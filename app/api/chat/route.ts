import { NextRequest, NextResponse } from 'next/server';
import { Message as VercelChatMessage, StreamingTextResponse } from 'ai';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { PineconeStore } from '@langchain/community/vectorstores/pinecone';
import { Document } from 'langchain/document';
import { RunnableSequence } from '@langchain/core/runnables';
import {
  BytesOutputParser,
  StringOutputParser,
} from '@langchain/core/output_parsers';
import { TogetherAIEmbeddings } from '@langchain/community/embeddings/togetherai';
import { Pinecone } from '@pinecone-database/pinecone';

export const runtime = 'edge';

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY ?? '',
  environment: process.env.PINECONE_ENVIRONMENT ?? '', //this is in the dashboard
});

const combineDocumentsFn = (docs: Document[], separator = '\n\n') => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join(separator);
};

const formatVercelMessages = (message: VercelChatMessage) => {
  if (message.role === 'user') {
    return `Human: ${message.content}`;
  } else if (message.role === 'assistant') {
    return `Assistant: ${message.content}`;
  } else {
    return `${message.role}: ${message.content}`;
  }
};

const CONDENSE_QUESTION_TEMPLATE = `Formuliere angesichts des folgenden Gesprächs und einer Folgefrage die Folgefrage so um, dass sie eine eigenständige Frage ist.

Chat-Verlauf:
<chat_history>
{chat_history}
</chat_history>
Follow Up Input: {question}
Eigenständige Frage:`;


const condenseQuestionPrompt = PromptTemplate.fromTemplate(
  CONDENSE_QUESTION_TEMPLATE,
);
const ANSWER_TEMPLATE = `Du bist ein hilfreicher KI-Assistent für die FN, die Deutsche Reiterliche Vereinigung und antwortest Fragen zum offiziellen Regelwerk der LPO (Leistungsprüfungsordnung). Beantworte die Frage am Ende anhand der folgenden Kontextinformationen.
Wenn du die Antwort nicht weißt, sage einfach, dass du es nicht weißt. Versuche NICHT, dir eine Antwort auszudenken.
Wenn die Frage nicht mit dem Kontext zusammenhängt, antworte höflich, dass du darauf eingestellt bist, nur Fragen zu beantworten, die mit dem Regelwerk zusammenhängen.

<context>
{context}
</context>

<chat_history>
  {chat_history}
</chat_history>

Question: {question}
Bitte sende eine Antwort in Markdown mit klaren Überschriften und Listen:`;

const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE);

/**
 * This handler initializes and calls a retrieval chain. It composes the chain using
 * LangChain Expression Language. See the docs for more information:
 *
 * https://js.langchain.com/docs/guides/expression_language/cookbook#conversational-retrieval-chain
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const formattedPreviousMessages = messages
      .slice(0, -1)
      .map(formatVercelMessages);
    const currentMessageContent = messages[messages.length - 1].content;
    const chatId = body.chatId;

    const model = new ChatOpenAI({
      modelName: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
      temperature: 0,
      configuration: {
        apiKey: process.env.TOGETHER_AI_API_KEY,
        baseURL: 'https://api.together.xyz/v1',
      },
    });

    const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME ?? '';
    const index = pinecone.index(PINECONE_INDEX_NAME);

    const vectorstore = await PineconeStore.fromExistingIndex(
      new TogetherAIEmbeddings({
        apiKey: process.env.TOGETHER_AI_API_KEY,
        modelName: 'togethercomputer/m2-bert-80M-8k-retrieval',
      }),
      {
        pineconeIndex: index,
        namespace: chatId,
      },
    );

    /**
     * We use LangChain Expression Language to compose two chains.
     * To learn more, see the guide here:
     *
     * https://js.langchain.com/docs/guides/expression_language/cookbook
     */
    const standaloneQuestionChain = RunnableSequence.from([
      condenseQuestionPrompt,
      model,
      new StringOutputParser(),
    ]);

    let resolveWithDocuments: (value: Document[]) => void;
    const documentPromise = new Promise<Document[]>((resolve) => {
      resolveWithDocuments = resolve;
    });

    const retriever = vectorstore.asRetriever({
      callbacks: [
        {
          handleRetrieverEnd(documents) {
            resolveWithDocuments(documents);
          },
        },
      ],
    });

    const retrievalChain = retriever.pipe(combineDocumentsFn as any);

    const answerChain = RunnableSequence.from([
      {
        context: RunnableSequence.from([
          (input) => input.question,
          retrievalChain,
        ]),
        chat_history: (input) => input.chat_history,
        question: (input) => input.question,
      },
      answerPrompt,
      model,
    ]);

    const conversationalRetrievalQAChain = RunnableSequence.from([
      {
        question: standaloneQuestionChain,
        chat_history: (input) => input.chat_history,
      },
      answerChain,
      new BytesOutputParser(),
    ]);

    const stream = await conversationalRetrievalQAChain.stream({
      chat_history: formattedPreviousMessages.join('\n'),
      question: currentMessageContent,
    });

    const documents = await documentPromise;
    const serializedSources = Buffer.from(
      JSON.stringify(
        documents.map((doc) => {
          return {
            pageContent: doc.pageContent.slice(0, 50) + '...',
            metadata: doc.metadata,
          };
        }),
      ),
    ).toString('base64');

    return new StreamingTextResponse(stream, {
      headers: {
        'x-message-index': (formattedPreviousMessages.length + 1).toString(),
        'x-sources': serializedSources,
      },
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
