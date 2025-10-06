import gradio as gr
import os
import numpy as np
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from docx import Document
import PyPDF2
from io import BytesIO

# Imports for utils
from utils.supabase_client import get_supabase
from utils.llm import get_llm
from utils.chunking import chunk_text
from utils.embeddings import embed_text
import hashlib  # For hashing full_text to cache questions

import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn

# Initialize clients from utils
supabase = get_supabase()
llm = get_llm()

# Custom CSS for modern HR style
CSS = """
body {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gradio-container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}
.panel {
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin: 10px 0;
}
.chat-message {
    background: #e3f2fd;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
}
"""

def extract_text_from_file(file):
    """Extract text from uploaded file (.txt, .docx, .pdf)"""
    if file is None:
        return ""
    try:
        if isinstance(file, str):
            # If it's a file path (string)
            with open(file, 'rb') as f:
                file_content = f.read()
            file_name = file
        else:
            # UploadedFile object
            file_content = file.read()
            file_name = file.name
        if file_name.endswith('.txt'):
            return file_content.decode('utf-8')
        elif file_name.endswith('.docx'):
            doc = Document(BytesIO(file_content))
            return '\n'.join([para.text for para in doc.paragraphs])
        elif file_name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        return ""
    except Exception as e:
        print(f"Error extracting text from {file_name}: {e}")
        return ""

# Use chunk_text from utils.chunking

# Use embed_text from utils.embeddings, which returns list of lists of floats

def store_embeddings(role_id, chunks, metadata_list):
    # Improvement: Batch generate embeddings for efficiency using utils.embed_text
    embeddings = embed_text(chunks)
    data = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
        data.append({
            'role_id': role_id,
            'chunk_text': chunk,
            'embedding': embeddings[i],
            'metadata': meta
        })
    supabase.table('embeddings').insert(data).execute()

def generate_role_questions(job_title, initial_summary, documents_text=""):
    full_text = initial_summary + "\n" + documents_text
    prompt = f"""
    Based on the job title '{job_title}' and the following role description (summary and documents): '{full_text[:2000]}...' (truncated if long), generate 5-8 specific clarification questions that a new employee might ask to understand the role better.
    Focus on ambiguities, processes, responsibilities, SOPs, frequencies, interactions, and details mentioned or implied in the text. Make questions targeted to elicit exact information, e.g., if summary mentions 'daily checks', ask 'What are the exact steps and criteria for daily checks?'.
    Questions should be relevant to the role and help clarify key aspects for onboarding. Do not use external knowledge; base questions only on the provided text to clarify ambiguities. Return as a numbered list.
    """
    # Improvement: Enhanced prompt to avoid external assumptions
    response = llm.invoke(prompt)
    questions = response.split('\n')
    questions = [q.strip() for q in questions if q.strip() and q[0].isdigit()]
    return questions[:8]  # Limit to 8

def generate_summary_without_qa(job_title, initial_summary, documents_text):
    full_text = initial_summary + "\n" + documents_text
    prompt = f"""
    Job Title: {job_title}
    Role Description (summary and documents): {full_text}
    
    Generate a comprehensive role summary based on the provided description. Make it concise yet detailed for a new employee, covering key responsibilities, processes, and details. Do not add external details; stick to provided information.
    """
    # Improvement: Enhanced prompt for strict adherence to input
    response = llm.invoke(prompt)
    return response

def generate_summary(job_title, initial_summary, qa_pairs):
    qa_text = '\n'.join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])
    prompt = f"""
    Job Title: {job_title}
    Initial Summary: {initial_summary}
    Clarification Q&A: {qa_text}
    
    Generate a comprehensive role summary incorporating the initial summary and clarifications.
    Make it concise yet detailed for a new employee. Do not add external details; stick to provided information.
    """
    # Improvement: Enhanced prompt for strict adherence to input
    response = llm.invoke(prompt)
    return response

# Outgoing interface state
outgoing_state = {
    'job_title': '',
    'initial_summary': '',
    'documents_text': '',
    'full_text_hash': '',  # Added for caching questions
    'questions': [],
    'answers': [],
    'role_id': None,
    'current_question_index': 0
}

def generate_summary_no_qa(job_title, initial_summary, files):
    if not job_title or not initial_summary:
        return gr.update(value=[{"role": "assistant", "content": "Please fill in Job Title and Summary."}]), gr.update(value="")
    
    # Extract full documents_text for LLM prompts
    documents_text = ''
    file_texts = []
    for file in files or []:
        text = extract_text_from_file(file)
        if text:
            documents_text += text + '\n'
            file_texts.append((file.name, text))

    full_text = initial_summary + '\n' + documents_text

    # Improvement: Chunk separately for accurate per-file metadata in embeddings, better for large docs and citations
    all_chunks = []
    all_metadata = []
    # Summary chunks
    summary_chunks = chunk_text(initial_summary)
    for sc in summary_chunks:
        all_chunks.append(sc)
        all_metadata.append({'source': 'initial_summary', 'type': 'summary'})
    # Document chunks per file
    for fname, ftext in file_texts:
        doc_chunks = chunk_text(ftext)
        for dc in doc_chunks:
            all_chunks.append(dc)
            all_metadata.append({'source': fname, 'type': 'document'})
    chunks = all_chunks
    metadata_list_full = all_metadata
    
    # Improvement: Reuse existing role if same job_title and initial_summary to avoid duplicates and enable updates (versioning via latest ID)
    res = supabase.table('roles').select('id').eq('job_title', job_title).eq('initial_summary', initial_summary).order('id', desc=True).limit(1).execute()
    if res.data:
        role_id = res.data[0]['id']
        # Delete old embeddings to update with current documents
        supabase.table('embeddings').delete().eq('role_id', role_id).execute()
    else:
        role_data = {
            'job_title': job_title,
            'initial_summary': initial_summary,
            'clarification_qa': []
        }
        res_insert = supabase.table('roles').insert(role_data).execute()
        role_id = res_insert.data[0]['id']

    # Store embeddings
    store_embeddings(role_id, chunks, metadata_list_full)

    # Generate summary without QA
    summary = generate_summary_without_qa(job_title, initial_summary, documents_text)

    # Update state for potential subsequent clarification session
    outgoing_state['role_id'] = role_id
    outgoing_state['job_title'] = job_title
    outgoing_state['initial_summary'] = initial_summary
    outgoing_state['documents_text'] = documents_text
    full_text_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()
    outgoing_state['full_text_hash'] = full_text_hash
    outgoing_state['questions'] = []
    outgoing_state['answers'] = []
    outgoing_state['current_question_index'] = 0
    
    # Save summary
    update_data = {
        'generated_summary': summary,
        'clarification_qa': []
    }
    supabase.table('roles').update(update_data).eq('id', role_id).execute()
    
    chat_history = [{"role": "assistant", "content": f"Role '{job_title}' saved. Generated Summary without clarification:\n{summary}"}]
    
    return gr.update(value=chat_history), gr.update(value="")

def start_outgoing_qa(job_title, initial_summary, files):
    if not job_title or not initial_summary:
        return gr.update(value=[{"role": "assistant", "content": "Please fill in Job Title and Summary."}]), gr.update(value="")
    
    # Extract full documents_text for LLM prompts
    documents_text = ''
    file_texts = []
    for file in files or []:
        text = extract_text_from_file(file)
        if text:
            documents_text += text + '\n'
            file_texts.append((file.name, text))

    full_text = initial_summary + '\n' + documents_text

    # Improvement: Chunk separately for accurate per-file metadata in embeddings, better for large docs and citations
    all_chunks = []
    all_metadata = []
    # Summary chunks
    summary_chunks = chunk_text(initial_summary)
    for sc in summary_chunks:
        all_chunks.append(sc)
        all_metadata.append({'source': 'initial_summary', 'type': 'summary'})
    # Document chunks per file
    for fname, ftext in file_texts:
        doc_chunks = chunk_text(ftext)
        for dc in doc_chunks:
            all_chunks.append(dc)
            all_metadata.append({'source': fname, 'type': 'document'})
    chunks = all_chunks
    metadata_list_full = all_metadata
    
    current_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()
    questions = []
    if outgoing_state['role_id'] and outgoing_state['job_title'] == job_title and outgoing_state['initial_summary'] == initial_summary and outgoing_state['full_text_hash'] == current_hash:
        role_id = outgoing_state['role_id']
        questions = outgoing_state['questions']
    else:
        # Improvement: Reuse existing role if same job_title and initial_summary to avoid duplicates and enable updates (versioning via latest ID)
        res = supabase.table('roles').select('id').eq('job_title', job_title).eq('initial_summary', initial_summary).order('id', desc=True).limit(1).execute()
        if res.data:
            role_id = res.data[0]['id']
            # Delete old embeddings to update with current documents
            supabase.table('embeddings').delete().eq('role_id', role_id).execute()
        else:
            role_data = {
                'job_title': job_title,
                'initial_summary': initial_summary,
                'clarification_qa': []
            }
            res_insert = supabase.table('roles').insert(role_data).execute()
            role_id = res_insert.data[0]['id']
    
    # Store embeddings with current content
    store_embeddings(role_id, chunks, metadata_list_full)
    
    # Generate questions if not cached
    if not questions:
        questions = generate_role_questions(job_title, initial_summary, documents_text)
    
    # Improvement: Cache questions in state if full_text unchanged (summary + documents) for consistency, regenerate only if changed
    
    # Update state
    outgoing_state['job_title'] = job_title
    outgoing_state['initial_summary'] = initial_summary
    outgoing_state['documents_text'] = documents_text
    outgoing_state['full_text_hash'] = current_hash
    outgoing_state['questions'] = questions
    outgoing_state['answers'] = []
    outgoing_state['role_id'] = role_id
    outgoing_state['current_question_index'] = 0
    
    welcome_msg = f"Welcome! Role '{job_title}' saved. Here are clarification questions:"
    chat_history = [{"role": "assistant", "content": welcome_msg}]
    if questions:
        chat_history.append({"role": "assistant", "content": questions[0]})
    
    return gr.update(value=chat_history), gr.update(value="")

def submit_outgoing_answer(answer, chat_history):
    chat_history = chat_history or []
    if outgoing_state['current_question_index'] < len(outgoing_state['questions']):
        q = outgoing_state['questions'][outgoing_state['current_question_index']]
        outgoing_state['answers'].append(answer)
        outgoing_state['current_question_index'] += 1
        
        chat_history.append({"role": "user", "content": answer})
        
        if outgoing_state['current_question_index'] < len(outgoing_state['questions']):
            next_q = outgoing_state['questions'][outgoing_state['current_question_index']]
            chat_history.append({"role": "assistant", "content": next_q})
        else:
            # All questions answered, generate summary
            qa_pairs = list(zip(outgoing_state['questions'], outgoing_state['answers']))
            summary = generate_summary(outgoing_state['job_title'], outgoing_state['initial_summary'], qa_pairs)
            # Save summary and QA
            update_data = {
                'generated_summary': summary,
                'clarification_qa': qa_pairs
            }
            supabase.table('roles').update(update_data).eq('id', outgoing_state['role_id']).execute()
            chat_history.append({"role": "assistant", "content": f"All questions answered! Generated Summary:\n{summary}"})
        
        return gr.update(value=chat_history), gr.update(value="")
    return gr.update(value=[{"role": "assistant", "content": "No more questions."}]), gr.update(value="")

# Incoming interface state
incoming_state = {'selected_role': ''}

def fetch_available_roles():
    res = supabase.table('roles').select('job_title').execute()
    return [r['job_title'] for r in res.data] if res.data else []

def load_incoming_summary(selected_role):
    if not selected_role:
        return "# No Role Selected\nPlease select your role."
    # Fetch from Supabase
    # Improvement: Fetch latest version by ID for role versioning
    res = supabase.table('roles').select('*').eq('job_title', selected_role).order('id', desc=True).limit(1).execute()
    if res.data:
        role = res.data[0]
        summary = role.get('generated_summary', role['initial_summary'])
        return f"# Role Summary for {selected_role}\n{summary}"
    return f"# No Summary Found for {selected_role}\nPlease wait for outgoing employee to provide data for this role."

def ask_incoming_question(question, selected_role, chat_history):
    chat_history = chat_history or []
    if not selected_role:
        chat_history.append({"role": "assistant", "content": "Please select your role first."})
        return gr.update(value=chat_history), gr.update(value="")
    
    # Fetch role_id first
    # Improvement: Fetch latest version by ID for role versioning
    role_res = supabase.table('roles').select('id').eq('job_title', selected_role).order('id', desc=True).limit(1).execute()
    if not role_res.data:
        chat_history.append({"role": "assistant", "content": "No data available for this role."})
        return gr.update(value=chat_history), gr.update(value="")
    role_id = role_res.data[0]['id']
    
    # RAG: Fetch embeddings for the role with metadata
    emb_res = supabase.table('embeddings').select('chunk_text, embedding, metadata').eq('role_id', role_id).execute()
    if not emb_res.data:
        chat_history.append({"role": "assistant", "content": "No detailed knowledge available for this role yet."})
        return gr.update(value=chat_history), gr.update(value="")
    
    chunks = [d['chunk_text'] for d in emb_res.data]
    metadatas = [d['metadata'] for d in emb_res.data]
    embeddings = []
    for d in emb_res.data:
        emb_str = d['embedding']
        if isinstance(emb_str, str):
            emb_list = json.loads(emb_str)
        else:
            emb_list = emb_str
        embeddings.append(np.array([float(val) for val in emb_list]))
    
    # Simple similarity search (top 25 chunks)
    # Improvement: Use embed_text for query embedding consistency with utils
    query_emb = np.array(embed_text([question])[0])
    similarities = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(embeddings)]
    top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:25]
    context = ''
    for idx in top_indices:
        i = idx[0]
        source = metadatas[i].get('source', 'unknown') if metadatas[i] else 'unknown'
        context += f"From {source}: {chunks[i]}\n"
    
    # Always run keyword fallback for additional recall
    question_keywords = question.lower().split()
    keyword_chunks = []
    for i, c in enumerate(chunks):
        if any(kw in c.lower() for kw in question_keywords):
            source = metadatas[i].get('source', 'unknown') if metadatas[i] else 'unknown'
            keyword_chunks.append(f"From {source}: {c}")
    if keyword_chunks:
        context += '\n\nAdditional keyword matches: ' + '\n'.join(keyword_chunks[:10])
    
    # Log for debugging
    print(f"Retrieved context for '{question}': {context[:500]}...")  # Truncated log
    
    prompt = f"""
    You are a strict role knowledge assistant. 
    Your job is to answer ONLY from the provided context. 

    CONTEXT:
    {context}

    QUESTION:
    {question}

    RULES:
    - Use ONLY what is inside CONTEXT. 
    - If the answer is not explicitly in the context, reply exactly: "Not specified in the provided documents."
    - Do NOT guess, infer, or use industry knowledge. 
    - Keep the answer factual, short, and tied to the source text.
    - Always mention the source file/section if available from metadata.
    - When mentioning a source file, format it exactly as [source:filename] where filename is the document name from the context (e.g., [source:HR.txt]).
    """

    response = llm.invoke(prompt)
    
    def make_sources_clickable(text):
        pattern = r'\[source:([^\]]+\.(txt|docx|pdf))\]'
        def replacer(match):
            filename = match.group(1)
            return f'<a href="/docs/{filename}" target="_blank" style="color: #007bff; text-decoration: underline;">{filename}</a>'
        return re.sub(pattern, replacer, text)
    
    response = make_sources_clickable(response)
    
    # Optional: Log interaction for analytics (assume interactions table exists: CREATE TABLE IF NOT EXISTS interactions (id SERIAL PRIMARY KEY, role_id INTEGER REFERENCES roles(id), question TEXT, answer TEXT, created_at TIMESTAMP DEFAULT NOW()); )
    # Improvement: Added optional interaction logging with error handling
    try:
        supabase.table('interactions').insert({
            'role_id': role_id,
            'question': question,
            'answer': response
        }).execute()
    except Exception as e:
        print(f"Failed to log interaction: {e}")
    
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": response})
    
    # Log interaction (optional, add table if needed)
    # supabase.table('interactions').insert({'role_id': role_id, 'question': question, 'answer': response}).execute()
    
    return gr.update(value=chat_history), gr.update(value="")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

app = FastAPI(title="Knowledge Transfer Agent")

@app.get("/docs/{filename}")
async def get_doc(filename: str):
    if not filename.endswith('.txt'):
        raise HTTPException(status_code=404, detail="File not found")
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/plain", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

# Gradio Blocks
with gr.Blocks(css=CSS, title="Knowledge Transfer Agent") as demo:
    gr.Markdown("# Knowledge Transfer Agent")
    
    with gr.Tabs():
        with gr.TabItem("Outgoing Employee"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Role Information Input")
                    job_title = gr.Textbox(label="Job Title", placeholder="e.g., Sales Representative")
                    initial_summary = gr.Textbox(label="Role Summary & Responsibilities", lines=6, placeholder="Describe your daily tasks, responsibilities...")
                    files = gr.File(label="Supporting Documents (Optional)", file_types=[".txt", ".docx", ".pdf"], file_count="multiple")
                    generate_summary_btn = gr.Button("Generate Summary Without Questions", variant="secondary")
                    start_qa_btn = gr.Button("Start Clarification Session", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Clarification Questions Chat")
                    outgoing_chat = gr.Chatbot(label="Chat History", height=400, type="messages")
                    outgoing_input = gr.Textbox(label="Answer the Question", placeholder="Type your answer here...")
                    submit_answer_btn = gr.Button("Submit Answer", variant="secondary")
            
            generate_summary_btn.click(
                generate_summary_no_qa,
                inputs=[job_title, initial_summary, files],
                outputs=[outgoing_chat, outgoing_input]
            )
            start_qa_btn.click(
                start_outgoing_qa,
                inputs=[job_title, initial_summary, files],
                outputs=[outgoing_chat, outgoing_input]
            )
            submit_answer_btn.click(
                submit_outgoing_answer,
                inputs=[outgoing_input, outgoing_chat],
                outputs=[outgoing_chat, outgoing_input]
            )
        
        with gr.TabItem("Incoming Employee"):
            gr.Markdown("### Select Your Role to View Summary")
            selected_role = gr.Dropdown(
                label="Your Job Title",
                choices=fetch_available_roles(),
                value=None,
                allow_custom_value=True
            )
            summary_display = gr.Markdown(label="Role Summary", visible=True)
            open_chat_btn = gr.Button("Open Q&A Chat", variant="primary")
            
            with gr.Row(visible=False) as chat_row:
                with gr.Column(scale=1):
                    gr.Markdown("### Role Summary")
                    summary_panel = gr.Markdown()
                with gr.Column(scale=1):
                    gr.Markdown("### Ask Questions About Your Role")
                    incoming_chat = gr.Chatbot(label="Q&A History", height=400, type="messages")
                    incoming_input = gr.Textbox(label="Your Question", placeholder="Ask something about your role...")
                    qna_submit_btn = gr.Button("Ask", variant="secondary")
            
            selected_role.change(load_incoming_summary, inputs=selected_role, outputs=summary_display)
            # Refresh choices on load or periodically if needed, but initial load suffices
            open_chat_btn.click(lambda: gr.update(visible=True), outputs=chat_row)
            open_chat_btn.click(lambda r: load_incoming_summary(r), inputs=selected_role, outputs=summary_panel)
            qna_submit_btn.click(
                ask_incoming_question,
                inputs=[incoming_input, selected_role, incoming_chat],
                outputs=[incoming_chat, incoming_input]
            )

if __name__ == "__main__":
    gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="127.0.0.1", port=7860)