from fastapi import HTTPException, Depends
from fastapi import UploadFile,File
from sqlalchemy.ext.asyncio  import AsyncSession 
from models import get_db , Chunk ,Document,Message,Conversation
from sqlalchemy import select
from fastapi.responses import StreamingResponse
from retrieval import chunk,generate_chunk_embedding
from crud import save_file,read_file
from schemas import Question
from retrieval import generate_conversation_title,hybrid_search,stream_prompt

async def upload_document_service (file:UploadFile = File(...),db:AsyncSession = Depends(get_db)):
    path = save_file(file)
    document = Document(
        filename=file.filename,
        file_path=str(path),
        file_type=file.content_type,
        file_size=file.size or 0
    )
    try:
        db.add(document)
        await db.flush()
        text = read_file(path)
        chunker = chunk(text)
        chunks = [c.strip() for c in chunker if c.strip()]
        embeddings = generate_chunk_embedding(chunks)
        for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
            db_chunk = Chunk(
                document_id=document.id,
                chunk_index=index,
                content=content,
                embedding=embedding,
            )
            db.add(db_chunk)
        await db.commit()
        await db.refresh(document)
        return document
    except Exception:
        await db.rollback()
        if path.exists():
            path.unlink() 
        raise
    


async def load_previous_messages(conversation_id:int,db:AsyncSession):
    history = ""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()
    for message in messages:
        role = message.role.capitalize()
        content = message.content
        history += f"{role}: {content}\n"
    return history

async def ask_question_service(question:Question,db:AsyncSession = Depends(get_db)):
    conversation = await db.get(
    Conversation,
    question.conversation_id,
)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    if not conversation.title:
        conversation.title = generate_conversation_title(
            question.text
        )
    await db.flush()
    history = await load_previous_messages(conversation_id=question.conversation_id, db=db)
    chunks = await hybrid_search(db=db,document_id=question.document_id,query=question.text)
    context = "\n\n".join(chunk.content for chunk in chunks)
    prompt = f"""
    You are answering questions about one uploaded document.
    Use ONLY the information contained in the context below.
    If the answer is present, answer it clearly.
    If the answer is not present, reply exactly:
    "I couldn't find that information in the uploaded document."
    History:{history}
    Context:{context}
    Question: {question.text} """
    return StreamingResponse(stream_prompt(prompt, db=db, conversation_id=question.conversation_id),media_type="text/plain")