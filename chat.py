import argparse
from dataclasses import asdict
import json
import os
import streamlit as st

from data_driven_characters.character import get_character_definition
from data_driven_characters.corpus import (
    get_corpus_summaries,
    load_docs,
)

from data_driven_characters.chatbots import (
    SummaryChatBot,
    RetrievalChatBot,
    SummaryRetrievalChatBot,
)
from data_driven_characters.interfaces import CommandLine, Streamlit

OUTPUT_ROOT = "output"


def create_chatbot(corpus, corpus2, character_name, character_name_2, chatbot_type, retrieval_docs, summary_type):
    # logging
    corpus_name = os.path.splitext(os.path.basename(corpus))[0]
    corpus_name_2 = os.path.splitext(os.path.basename(corpus2))[0]
    output_dir = f"{OUTPUT_ROOT}/{corpus_name}/summarytype_{summary_type}"
    output_dir_2 = f"{OUTPUT_ROOT}/{corpus_name_2}/summarytype_{summary_type}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)
    summaries_dir = f"{output_dir}/summaries"
    summaries_dir_2 = f"{output_dir_2}/summaries"
    character_definitions_dir = f"{output_dir}/character_definitions"
    character_definitions_dir_2 = f"{output_dir_2}/character_definitions"
    os.makedirs(character_definitions_dir, exist_ok=True)
    os.makedirs(character_definitions_dir_2, exist_ok=True)
    # load docs
    docs = load_docs(corpus_path=corpus, chunk_size=2048, chunk_overlap=64)
    docs_2 = load_docs(corpus_path=corpus2, chunk_size=2048, chunk_overlap=64)
    
    # generate summaries
    corpus_summaries = get_corpus_summaries(
        docs=docs, summary_type=summary_type, cache_dir=summaries_dir
    )
    corpus_summaries_2 = get_corpus_summaries(
        docs=docs_2, summary_type=summary_type, cache_dir=summaries_dir_2
    )

    # get character definition
    character_definition = get_character_definition(
        name=character_name,
        corpus_summaries=corpus_summaries,
        cache_dir=character_definitions_dir,
    )

    character_definition_2 = get_character_definition(
        name=character_name_2,
        corpus_summaries=corpus_summaries_2,
        cache_dir=character_definitions_dir,
    )
    print(json.dumps(asdict(character_definition), indent=4))
    print(json.dumps(asdict(character_definition_2), indent=4))
    # construct retrieval documents
    if retrieval_docs == "raw":
        documents = [
            doc.page_content
            for doc in load_docs(corpus_path=corpus, chunk_size=256, chunk_overlap=16)
        ]
    elif retrieval_docs == "summarized":
        documents = corpus_summaries
        documents_2 = corpus_summaries_2
    else:
        raise ValueError(f"Unknown retrieval docs type: {retrieval_docs}")

    # initialize chatbot
    if chatbot_type == "summary":
        chatbot = SummaryChatBot(character_definition=character_definition)
    elif chatbot_type == "retrieval":
        chatbot = RetrievalChatBot(
            character_definition=character_definition,
            character_definition_2=character_definition_2,
            documents=documents,
            documents_2=documents_2
        )
    elif chatbot_type == "summary_retrieval":
        chatbot = SummaryRetrievalChatBot(
            character_definition=character_definition,
            documents=documents,
        )
    else:
        raise ValueError(f"Unknown chatbot type: {chatbot_type}")
    return chatbot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=str, default="data/everything_everywhere_all_at_once.txt"
    )
    parser.add_argument(
        "--corpus2", type=str, default="data/thor_love_and_thunder.txt"
    )
    parser.add_argument("--character_name", type=str, default="Evelyn")
    parser.add_argument("--character_name_2", type=str, default="Thor")
    parser.add_argument(
        "--chatbot_type",
        type=str,
        default="summary_retrieval",
        choices=["summary", "retrieval", "summary_retrieval"],
    )
    parser.add_argument(
        "--summary_type",
        type=str,
        default="map_reduce",
        choices=["map_reduce", "refine"],
    )
    parser.add_argument(
        "--retrieval_docs",
        type=str,
        default="summarized",
        choices=["raw", "summarized"],
    )
    parser.add_argument(
        "--interface", type=str, default="cli", choices=["cli", "streamlit"]
    )
    args = parser.parse_args()

    if args.interface == "cli":
        chatbot = create_chatbot(
            args.corpus,
            args.character_name,
            args.character_name_2,
            args.chatbot_type,
            args.retrieval_docs,
            args.summary_type,
        )
        app = CommandLine(chatbot=chatbot)
    elif args.interface == "streamlit":
        chatbot = st.cache_resource(create_chatbot)(
            args.corpus,
            args.corpus2,
            args.character_name,
            args.character_name_2,
            args.chatbot_type,
            args.retrieval_docs,
            args.summary_type,
        )
        st.title("AI SIMS")
        st.write("Create your own sims groupchats, grounded in existing corpora.")
        st.divider()
        st.markdown(f"**chatbot type**: *{args.chatbot_type}*")
        if "retrieval" in args.chatbot_type:
            st.markdown(f"**retrieving from**: *{args.retrieval_docs} corpus*")
        app = Streamlit(chatbot=chatbot)
    else:
        raise ValueError(f"Unknown interface: {args.interface}")
    app.run()


if __name__ == "__main__":
    main()
