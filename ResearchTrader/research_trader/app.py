"""
Gradio frontend for ResearchTrader
"""

import os
from datetime import datetime
from typing import Any

import gradio as gr
import httpx
import pandas as pd

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEMO_TITLE = "ResearchTrader"
DEMO_DESCRIPTION = """
# ResearchTrader: Quantitative Finance Research Explorer

Discover, explore, and operationalize the latest ArXiv quantitative finance research.
Search for papers, explore structured summaries, ask questions, and generate Python trading strategies.
"""
DEMO_ARTICLE = """
## How to Use
1. **Search**: Enter keywords to find relevant q-fin papers
2. **Explore**: Click on a paper to view its structured summary
3. **Ask**: Use the Q&A feature to ask questions about the paper
4. **Generate**: Create Python trading strategies based on selected papers
"""


# API Client
async def search_papers(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search for papers"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/search/", params={"query": query, "max_results": max_results}
        )
        if response.status_code == 200:
            return response.json()["papers"]
        else:
            return []


async def get_paper_structure(paper_id: str) -> dict[str, str]:
    """Get paper structure"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/structure/{paper_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "objective": "Error fetching structure",
                "methods": "Error fetching structure",
                "results": "Error fetching structure",
                "conclusions": "Error fetching structure",
            }


async def get_paper_summary(paper_id: str) -> dict[str, Any]:
    """Get paper summary"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/summarize/{paper_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"summary": "Error fetching summary"}


async def ask_question(question: str, paper_ids: list[str]) -> str:
    """Ask a question about papers"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/qa/", json={"question": question, "paper_ids": paper_ids}
        )
        if response.status_code == 200:
            result = response.json()
            return f"{result['answer']}\n\nConfidence: {result['confidence']:.2f}"
        else:
            return f"Error: {response.text}"


async def generate_strategy(
    paper_ids: list[str], market: str, timeframe: str, risk_profile: str, additional_context: str
) -> str:
    """Generate a trading strategy"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/strategy/",
            json={
                "paper_ids": paper_ids,
                "market": market,
                "timeframe": timeframe,
                "risk_profile": risk_profile,
                "additional_context": additional_context,
            },
        )
        if response.status_code == 200:
            result = response.json()
            return f"# {result['strategy_name']}\n\n{result['description']}\n\n```python\n{result['python_code']}\n```\n\n## Usage Notes\n{result['usage_notes']}\n\n## Limitations\n{result['limitations']}"
        else:
            return f"Error: {response.text}"


# Gradio UI Functions
def format_paper_list(papers: list[dict[str, Any]]) -> pd.DataFrame:
    """Format papers for display in a dataframe"""
    formatted_papers = []
    for p in papers:
        # Format date
        pub_date = datetime.fromisoformat(p["published"].replace("Z", "+00:00"))
        formatted_date = pub_date.strftime("%Y-%m-%d")

        # Format authors (limit to 3)
        authors = p["authors"]
        if len(authors) > 3:
            author_str = ", ".join(authors[:3]) + f" +{len(authors) - 3}"
        else:
            author_str = ", ".join(authors)

        # Format summary (truncate)
        summary = p["summary"]
        if len(summary) > 200:
            summary = summary[:197] + "..."

        formatted_papers.append(
            {
                "ID": p["id"],
                "Title": p["title"],
                "Authors": author_str,
                "Date": formatted_date,
                "Category": p["category"],
                "Summary": summary,
            }
        )

    return pd.DataFrame(formatted_papers)


# Gradio UI Components
async def search_and_display(query, max_results):
    """Search for papers and display results"""
    if not query.strip():
        return None, gr.update(visible=False), None

    papers = await search_papers(query, max_results)
    if not papers:
        return None, gr.update(visible=True, value="No papers found matching your query."), None

    df = format_paper_list(papers)
    paper_ids = [p["id"] for p in papers]

    return df, gr.update(visible=False), paper_ids


async def load_paper_details(evt: gr.SelectData, papers_df, paper_ids):
    """Load paper details when a row is selected"""
    selected_idx = evt.index[0]
    if selected_idx >= len(paper_ids):
        return None, None, None, None

    paper_id = paper_ids[selected_idx]

    # Fetch paper structure and summary in parallel
    structure_task = get_paper_structure(paper_id)
    summary_task = get_paper_summary(paper_id)

    structure = await structure_task
    summary = await summary_task

    paper_title = papers_df.iloc[selected_idx]["Title"]

    return (
        f"# {paper_title}\n\n{summary.get('summary', 'Summary not available')}",
        structure.get("objective", "Not available"),
        structure.get("methods", "Not available"),
        structure.get("results", "Not available"),
        structure.get("conclusions", "Not available"),
        paper_id,
    )


async def ask_question_about_paper(question, selected_paper_id):
    """Ask a question about the selected paper"""
    if not question.strip() or not selected_paper_id:
        return "Please select a paper and enter a question."

    return await ask_question(question, [selected_paper_id])


async def generate_paper_strategy(
    selected_paper_id, market, timeframe, risk_profile, additional_context
):
    """Generate a strategy based on the selected paper"""
    if not selected_paper_id:
        return "Please select a paper first."

    return await generate_strategy(
        [selected_paper_id], market, timeframe, risk_profile, additional_context
    )


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(DEMO_DESCRIPTION)

    # Hidden state variables
    paper_ids = gr.State([])
    selected_paper_id = gr.State(None)

    with gr.Tab("Search"):
        with gr.Row():
            with gr.Column(scale=4):
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter keywords to search for q-fin papers...",
                    lines=1,
                )
            with gr.Column(scale=1):
                max_results = gr.Slider(
                    minimum=5, maximum=20, value=10, step=1, label="Max Results"
                )
            with gr.Column(scale=1):
                search_button = gr.Button("Search", variant="primary")

        no_results = gr.Markdown(visible=False)
        papers_df = gr.Dataframe(
            headers=["Title", "Authors", "Date", "Category", "Summary"],
            interactive=False,
            wrap=True,
        )

        search_button.click(
            search_and_display,
            inputs=[search_input, max_results],
            outputs=[papers_df, no_results, paper_ids],
        )

    with gr.Tab("Paper Details"):
        with gr.Row():
            with gr.Column():
                summary_md = gr.Markdown(label="Summary")

        with gr.Tabs():
            with gr.TabItem("Objective"):
                objective_md = gr.Markdown()
            with gr.TabItem("Methods"):
                methods_md = gr.Markdown()
            with gr.TabItem("Results"):
                results_md = gr.Markdown()
            with gr.TabItem("Conclusions"):
                conclusions_md = gr.Markdown()

    with gr.Tab("Ask Questions"):
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about the selected paper...",
                    lines=2,
                )
                question_button = gr.Button("Ask", variant="primary")

        answer_output = gr.Markdown(label="Answer")

        question_button.click(
            ask_question_about_paper,
            inputs=[question_input, selected_paper_id],
            outputs=answer_output,
        )

    with gr.Tab("Generate Strategy"):
        with gr.Row():
            with gr.Column():
                market = gr.Dropdown(
                    ["equities", "forex", "crypto", "futures"], label="Market", value="equities"
                )
                timeframe = gr.Dropdown(
                    ["tick", "minute", "hourly", "daily", "weekly"],
                    label="Timeframe",
                    value="daily",
                )
                risk_profile = gr.Dropdown(
                    ["conservative", "moderate", "aggressive"],
                    label="Risk Profile",
                    value="moderate",
                )
                additional_context = gr.Textbox(
                    label="Additional Context",
                    placeholder="Add any specific requirements or constraints...",
                    lines=3,
                )
                generate_button = gr.Button("Generate Strategy", variant="primary")

        strategy_output = gr.Markdown(label="Generated Strategy")

        generate_button.click(
            generate_paper_strategy,
            inputs=[selected_paper_id, market, timeframe, risk_profile, additional_context],
            outputs=strategy_output,
        )

    # Connect the paper selection to details tab
    papers_df.select(
        load_paper_details,
        inputs=[papers_df, paper_ids],
        outputs=[
            summary_md,
            objective_md,
            methods_md,
            results_md,
            conclusions_md,
            selected_paper_id,
        ],
    )

    gr.Markdown(DEMO_ARTICLE)

if __name__ == "__main__":
    demo.launch()
