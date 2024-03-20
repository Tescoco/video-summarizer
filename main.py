from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse
import moviepy.editor as mp 
import speech_recognition as sr 
import replicate
from typing import List

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as file:
        html_content = file.read()
    return Response(content=html_content, media_type="text/html")

@app.post("/upload-video/")
async def upload_video(video: UploadFile = File(...)):
    
    with open("video.mp4", "wb") as f:
        f.write(await video.read())
    
    video_file = mp.VideoFileClip("video.mp4")
    audio_file = video_file.audio 
    audio_file.write_audiofile("video.wav") 

    r = sr.Recognizer() 

    with sr.AudioFile("video.wav") as source: 
        data = r.record(source)

    text = r.recognize_google(data)
    
    summary_list: List[str] = []
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt = "give the summary of the video with the audio as follows: " + text
    
    for event in replicate.stream(
      "meta/llama-2-70b-chat",
       input={
         "debug": False,
         "top_p": 1,
         "prompt": prompt,
         "temperature": 0.5,
         "system_prompt": system_prompt,
         "max_new_tokens": 500,
         "min_new_tokens": -1,
         "prompt_template": "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
         "repetition_penalty": 1.15
       },
    ):
    
      summary = str(event)
      summary_list.append(summary)
    full_summary = "\n".join(summary_list).replace("\n", " ")
     
    return {"summary": full_summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

