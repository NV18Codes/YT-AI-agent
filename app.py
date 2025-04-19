import streamlit as st
from streamlit_lottie import st_lottie
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import emoji
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import openai
import google.generativeai as genai
from dotenv import load_dotenv
from youtube_comment_downloader import YoutubeCommentDownloader
from yt_dlp import YoutubeDL

# --- Load Keys ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Language Configuration ---
allowed_languages = ["en", "hi", "kn", "te", "ta", "ko", "ja"]

# --- Language Detection ---
def safe_detect_language(text):
    try:
        if not text or len(text.strip()) < 3:
            return "en"
        if all(char in emoji.EMOJI_DATA for char in text.strip()):
            return "emoji"

        transliterated_keywords = {
            "hi": ["kon", "kya", "kaise", "video", "dekh", "pyar", "mera", "bhot", "acha", "dil"],
            "ta": ["ungal", "pudhu", "nalla", "enna", "romba"],
            "te": ["chala", "video", "manchi", "meeru"],
            "kn": ["hesaru", "ondhu", "nodi"]
        }
        text_lower = text.lower()
        for lang, keywords in transliterated_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return lang

        lang = detect(text)
        return lang if lang in allowed_languages else "en"
    except LangDetectException:
        return "en"

# --- Tone Detection ---
def detect_tone(text):
    if not text:
        return "Neutral"
    text_lower = text.lower()

    if all(char in emoji.EMOJI_DATA for char in text):
        return "Praise"
    if re.search(r"\b\d{1,2}:\d{2}\b", text_lower):
        return "Timestamp"
    if re.search(r"\b(19|20)\d{2}\b", text_lower):
        return "Year"
    if any(word in text_lower for word in ["lol", "lmao", "rofl", "haha"]):
        return "Humor"

    positive_emojis = ["😂", "😍", "❤️", "😊", "😁", "🥰", "🎉", "🖤", "💖", "✨"]
    if any(e in text for e in positive_emojis):
        return "Praise"

    patterns = {
        "Praise": ["love", "beautiful", "amazing", "nice", "great", "wonderful", "awesome"],
        "Criticism": ["bad", "worst", "hate", "cringe", "trash", "dislike"],
        "Support": ["respect", "keep it up", "well done", "good job"],
        "Confusion": ["what", "why", "confused"],
        "Request": ["please", "can you", "would you", "could you"]
    }

    for tone, keywords in patterns.items():
        if any(word in text_lower for word in keywords):
            return tone

    return "Neutral"

# --- Sentiment Detection ---
def detect_sentiment(text):
    return "Positive" if "good" in text.lower() else "Neutral"

# --- Translation ---
def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except:
        return text

def translate_back(reply, lang):
    try:
        if lang != "en" and lang in allowed_languages:
            return GoogleTranslator(source="en", target=lang).translate(reply)
        return reply
    except:
        return reply

# --- AI Reply with Fallback ---
def generate_reply(comment, lang="en"):
    if not comment:
        return "Thanks for your feedback! 👍"
    try:
        return translate_back(openai_reply(comment), lang)
    except Exception as e:
        if "quota" in str(e).lower():
            try:
                return translate_back(gemini_reply(comment), lang)
            except:
                return translate_back(llama_reply(comment), lang)
        return translate_back("Thanks for your feedback! 👍", lang)

def openai_reply(comment):
    prompt = f"You are a friendly AI replying to YouTube comments. Be concise.\nComment: {comment}\nReply:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def gemini_reply(comment):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(f"Reply concisely to this YouTube comment:\n\n{comment}")
    return response.text.strip()

def llama_reply(comment):
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": f"Reply concisely to this YouTube comment:\n\n{comment}"
        }, timeout=20)
        return r.json().get("response", "Thanks for your feedback! 👍").strip()
    except:
        return "Thanks for your feedback! 👍"

# --- Video Info & Summary ---
def get_video_info(url):
    try:
        with YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get("title", "Untitled")
        genre = info.get("categories", ["Unknown"])
        description = info.get("description", "")
        try:
            summary = gemini_reply(f"Summarize this video description in 2-3 lines:\n{description}")
        except:
            summary = openai_reply(f"Summarize this video description in 2-3 lines:\n{description}")

        children_keywords = ["kids", "nursery", "rhymes", "baby", "child", "toddlers", "cartoon", "children", "learning"]
        children_friendly = any(k in title.lower() or k in description.lower() for k in children_keywords)

        return title, summary, genre[0], "Yes" if children_friendly else "No"
    except Exception as e:
        return "Unknown", f"Could not summarize: {e}", "Unknown", "Unknown"

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="YouTube Comment AI Replier", layout="wide", page_icon="🤖")
    st.title("🤖 YouTube Comment AI Replier")
    st.markdown("💬 Automatically detect language, analyze tone, and generate AI replies with OpenAI, Gemini, and LLaMA fallback!")

    url = st.text_input("📺 Enter YouTube video URL")
    count = st.slider("Number of comments to fetch", 5, 50, 10)
    show_charts = st.checkbox("Show analytics", True)

    if st.button("✨ Generate Replies") and url:
        title, summary, genre, children = get_video_info(url)
        st.subheader("🎥 Video Overview")
        st.markdown(f"**Title:** {title}")
        st.markdown(f"**Summary:** {summary}")
        st.markdown(f"**Genre:** {genre}")
        st.markdown(f"**Children Friendly:** {'✅ Yes' if children == 'Yes' else '❌ No'}")

        st.info("⏳ Processing comments and generating replies...")
        downloader = YoutubeCommentDownloader()
        comments = [c["text"] for _, c in zip(range(count), downloader.get_comments_from_url(url))]

        data = []
        for comment in comments:
            lang = safe_detect_language(comment)
            translated = translate_to_english(comment) if lang not in ["en", "emoji"] else comment

            tone = detect_tone(translated)
            sentiment = detect_sentiment(translated)
            reply = generate_reply(translated, lang=lang)

            data.append({
                "Original Comment": comment,
                "Reply": reply,
                "Detected Language": lang,
                "Tone": tone,
                "Sentiment": sentiment
            })

        df = pd.DataFrame(data)
        st.subheader("💬 AI Generated Replies")
        st.dataframe(df)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "ai_replies.csv")

        if show_charts:
            st.subheader("📊 Analytics")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="Sentiment", palette="viridis", ax=ax)
                ax.set_title("Sentiment Distribution")
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="Tone", palette="coolwarm", ax=ax)
                ax.set_title("Tone Analysis")
                st.pyplot(fig)

if __name__ == "__main__":
    main()
