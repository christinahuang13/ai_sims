import faiss
from tqdm import tqdm

from langchain.chains import ConversationChain
from langchain_anthropic import ChatAnthropic
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS

from data_driven_characters.memory import ConversationVectorStoreRetrieverMemory


class RetrievalChatBot:
    def __init__(self, character_definition, character_definition_2, documents):
        print("TESTINGSTRING123")
        print(character_definition)
        self.character_definition = character_definition
        self.character_definition_2 = character_definition_2
        print("PRINTING NEW DEFINITION")
        print(character_definition_2)
        self.documents = documents
        self.num_context_memories = 10
        # true if character1, false if character2
        self.current = true
        self.chat_history_key = "chat_history"
        self.context_key = "context"
        self.input_key = "input"

        self.chain_1 = self.create_chain(character_definition, character_definition_2)
        self.chain_2 = self.create_chain(character_definition_2, character_definition)



    def create_chain(self, character_definition, character_definition_2):

        conv_memory = ConversationBufferMemory(
            memory_key=self.chat_history_key, input_key=self.input_key
        )

        context_memory = ConversationVectorStoreRetrieverMemory(
            retriever=FAISS(
                OpenAIEmbeddings().embed_query,
                faiss.IndexFlatL2(1536),  # Dimensions of the OpenAIEmbeddings
                InMemoryDocstore({}),
                {},
            ).as_retriever(search_kwargs=dict(k=self.num_context_memories)),
            memory_key=self.context_key,
            output_prefix=character_definition.name,
            blacklist=[self.chat_history_key],
        )
        # add the documents to the context memory
        for i, summary in tqdm(enumerate(self.documents)):
            context_memory.save_context(inputs={}, outputs={f"[{i}]": summary})

        # Combined
        memory = CombinedMemory(memories=[conv_memory, context_memory])
        prompt = PromptTemplate.from_template(
            f"""Your name is {character_definition.name}.


You will have a conversation with another fictional character, {character_definition_2.name}, and you will engage in a dialogue with them.
You will exaggerate your personality, interests, desires, emotions, and other traits.
You will stay in character as {character_definition.name} throughout the conversation, even if the {character_definition_2.name} asks you questions that you don't know the answer to.
You will not break character as {character_definition.name}.

You are {character_definition.name} in the following story snippets, which describe events in your life.
---
{{{self.context_key}}}
---

Current conversation:
---
{character_definition.name}: {character_definition.greeting}
{{{self.chat_history_key}}}
---

{character_definition_2.name}: {character_definition_2.greeting}
{character_definition.name}:"""
        )
        GPT3 = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
        chatbot = ConversationChain(
            llm=GPT3, verbose=True, memory=memory, prompt=prompt
        )
        return chatbot

    def greet(self):
        return self.character_definition.greeting

    def step(self, input, is_user=True):
        if not is_user:
            self.current = not self.current
            # change to last convo
            return self.chain_1.run(input=input) if self.current else self.chain_2.run(input=input)

        return self.chain_1.run(input=input) if self.current else self.chain_2.run(input=input)
