# pylint: disable=missing-module-docstring, missing-class-docstring, line-too-long, missing-function-docstring
# pyright: reportCallIssue=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingTypeStubs=false
# pyright: reportArgumentType=false

import json
import logging
import os
import sys
from typing import cast

import streamlit as st
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.retrievers.bm25 import BM25Retriever

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


HF_TOKEN = os.getenv("HF_TOKEN")

st.header("Chat with the Fusick Automotive products bot! 🚗 🚙 🔧 🔩")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything about Fusick Automotive catalog # 57C! 👨‍🔧",
        },
    ]


class HybridRetriever(BaseRetriever):
    def __init__(
        self, vector_retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        bm25_nodes: list[NodeWithScore] = self.bm25_retriever.retrieve(query_bundle)
        vector_nodes: list[NodeWithScore] = self.vector_retriever.retrieve(query_bundle)

        all_nodes: list[NodeWithScore] = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


TEXT_QA_TEMPLATE = (
    "You are a bot specializing in browsing the auto parts catalogue of Fusick Automotive Products.\n"
    "Fusick Automotive Products is one of the largest manufacturers and suppliers of classic and antique automobile restoration parts.\n"
    "They have specialized in Oldsmobile parts for the 88, 98, Toronado, Cutlass, 442, and Hurst Olds since 1971.\n"
    "You are given products from the catalogue with highest similarity to the user query as context below.\n\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Given this information, please answer the user query: {query_str}\n"
    "Give your response in a concise format including page number and code for each item.\n"
    "Include the price of the item if it is mentioned in the context.\n"
    "Use hyphens, colons and newlines when displaying the results.\n"
    "Answer:"
)


@st.cache_resource(show_spinner=False)
def create_engine():
    with st.spinner(text="Creating a hybrid query engine..."):
        documents = SimpleDirectoryReader("data").load_data()

        nodes: list[TextNode] = []

        for doc in documents:
            content = doc.to_embedchain_format()["data"]["content"]
            for line in content.splitlines():
                parsed_line = json.loads(line)
                code = parsed_line["code"]
                description = parsed_line["description"]
                page = parsed_line["page"]
                node = TextNode(text=description, metadata={"code": code, "page": page})
                nodes.append(node)

        local_embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5",
        )

        remote_llm = HuggingFaceInferenceAPI(
            model_name="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN
        )

        Settings.embed_model = local_embed_model
        Settings.llm = remote_llm

        index = VectorStoreIndex(nodes=nodes, show_progress=True)

        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
        )

        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

        text_qa_template = PromptTemplate(TEXT_QA_TEMPLATE)

        response_synthesizer = get_response_synthesizer(
            text_qa_template=text_qa_template,
        )

        hybrid_query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )

        return hybrid_query_engine


query_engine = create_engine()


if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = cast(PydanticResponse, query_engine.query(prompt))
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
