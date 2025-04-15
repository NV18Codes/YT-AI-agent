import streamlit as st
from streamlit_lottie import st_lottie
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import emoji
import re
from transformers import pipeline, set_seed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
from dotenv import load_dotenv
from youtube_comment_downloader import YoutubeCommentDownloader
from yt_dlp import YoutubeDL
import requests

# --- Setup ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
set_seed(42)
sentiment_analyzer = pipeline("sentiment-analysis", device=-1)

# --- Allowed Languages ---
allowed_languages = ['en', 'hi', 'kn', 'te', 'ta', 'ko', 'ja']

# --- Utilities ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def safe_detect_language(text):
    try:
        if not text or len(text.strip()) < 3:
            return 'en'

        # Check for emoji-only text
        if all(char in emoji.EMOJI_DATA for char in text.strip()):
            return 'emoji'

        # Manually checking for language patterns (for Indian languages like Kannada, Telugu, Tamil)
        # Simple keywords to detect these languages
        indian_keywords = {
            'hi': ['kaise', 'kya', 'aap', 'hai', 'kar', 'ho', 'pyar', 'dost'],
            'kn': ['ಹೇಗಿದೆ', 'ನೀವು', 'ಅವನು', 'ಇದು', 'ಪ್ರೀತಿ'],
            'te': ['ఏంటి', 'నేను', 'మీరు', 'అది', 'ప్రేమ'],
            'ta': ['எப்படி', 'நீங்கள்', 'அது', 'காதல்', 'பொறுப்பு'],
            'ko': ['어떻게', '너', '그것', '사랑'],
            'ja': ['どう', 'あなた', 'それ', '愛'],
        }
        
        # Check if the comment contains keywords from any language
        for lang, keywords in indian_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                return lang
        
        # Use langdetect to detect the language
        lang = detect(text)
        if lang in allowed_languages:
            return lang
        return 'en'
    
    except LangDetectException:
        return 'en'

def detect_tone(text):
    if not text:
        return "Neutral"

    text_lower = text.lower().strip()
    if all(char in emoji.EMOJI_DATA for char in text.strip()):
        return "Emoji"

    if re.search(r"\b\d{1,2}:\d{2}\b", text_lower):
        return "Timestamp"

    if re.search(r"\b(19|20)\d{2}\b", text_lower):
        return "Year"

    patterns = {
        "Humor": r"\b(lol|lmao|rofl|haha|funny)\b",
        "Praise": r"\b(love|amazing|great job|thank you|awesome|best)\b",
        "Criticism": r"\b(bad|worst|hate|trash|dislike)\b",
        "Support": r"\b(nice work|keep it up|respect|well done)\b",
        "Confusion": r"\b(confused|what)\b",
        "Request": r"\b(can you|please|would you|could you)\b",
        "Disapproval": r"\b(wtf|cringe|eww)\b",
    }

    for tone, pattern in patterns.items():
        if re.search(pattern, text_lower):
            return tone
    return "Neutral"

def detect_sentiment(text):
    try:
        if not text:
            return "Neutral"
        result = sentiment_analyzer(text)[0]
        return result['label'].capitalize()
    except:
        return "Neutral"

def translate_to_english(text):
    try:
        if not text:
            return text
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def translate_back(reply, lang):
    try:
        if lang in allowed_languages and lang != 'en':
            return GoogleTranslator(source='en', target=lang).translate(reply)
        return reply
    except:
        return reply

def generate_reply(comment, model="gpt-4", video_title=""):
    if not comment:
        return "Thanks for your feedback! 👍"
    
    tone = detect_tone(comment)

    if tone == "Emoji":
        return comment
    if tone == "Timestamp":
        return "IKR! 🔥 That part was epic!"
    if tone == "Year":
        return "Good music days! 🎶 Let's vibe!"

    tone_templates = {
        "Humor": "😂 That was funny! Glad you're enjoying it!",
        "Criticism": "Thanks for your feedback. We'll improve! 🙏",
        "Praise": "So glad you liked it! 😊",
        "Support": "Appreciate the support! 💪",
        "Confusion": "Sorry if it wasn't clear. Let us know! 🤔",
        "Request": "Thanks for the suggestion! We'll consider it. 🙌",
        "Disapproval": "We'll try to do better next time. 💡",
    }

    if tone in tone_templates:
        return tone_templates[tone]

    try:
        prompt = f"""You are a helpful, friendly AI that replies to YouTube comments.
Reply in 1-2 conversational sentences. Include a light emoji if appropriate.

Comment: \"{comment}\"
Reply:"""
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        reply = response.choices[0].message['content'].strip()
        return emoji.emojize(reply, language='alias')
    except:
        return "Thanks for your feedback! 👍"

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="YouTube Comment AI Replier", layout="wide", page_icon="🤖")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""<h1 style='font-size: 2.7rem;'>🤖 YouTube Comment <span style='color:#0059b3;'>AI Replier</span></h1>
            <p style='font-size: 1.2rem;'>✨ Craft replies and understand viewer sentiment instantly!</p>
        """, unsafe_allow_html=True)
    with col2:
        lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_w51pcehl.json")
        if lottie:
            st_lottie(lottie, height=170, key="ai")

    st.sidebar.title("⚙️ Configuration")
    model_choice = st.sidebar.selectbox("🤖 Model", ["gpt-4"])
    comment_count = st.sidebar.slider("💬 Number of Comments", 5, 50, 10)
    show_charts = st.sidebar.checkbox("📊 Show Charts", value=True)

    url = st.text_input("📽️ YouTube Video URL or ID")
    sort_by = st.selectbox("🔽 Sort Results By", ["Original", "Tone", "Sentiment"])

    if st.button("✨ Generate Replies") and url:
        try:
            st.info("Fetching video info and comments...")
            ydl_opts = {"quiet": True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get("title") or "Untitled"

            downloader = YoutubeCommentDownloader()
            comments = [c['text'] for _, c in zip(range(comment_count), downloader.get_comments_from_url(url))]

            replies, tones, sentiments, langs = [], [], [], []

            with st.spinner("Generating replies..."):
                for c in comments:
                    lang = safe_detect_language(c)
                    langs.append(lang)

                    if lang in ['en', 'emoji']:
                        eng_comment = c
                    else:
                        eng_comment = translate_to_english(c)

                    tone = detect_tone(eng_comment)
                    sentiment = detect_sentiment(eng_comment)
                    reply = generate_reply(eng_comment, model=model_choice, video_title=video_title)

                    final_reply = translate_back(reply, lang)
                    replies.append(final_reply)
                    tones.append(tone)
                    sentiments.append(sentiment)

            # Here we maintain the original order
            df = pd.DataFrame({
                "Comment": comments,
                "Reply": replies,
                "Tone": tones,
                "Sentiment": sentiments,
                "Lang": langs
            })

            if sort_by != "Original":
                df = df.sort_values(by=sort_by)

            st.subheader("💬 AI Replies")
            st.dataframe(df)

            st.download_button("📥 Download CSV", df.to_csv(index=False), "youtube_replies.csv", "text/csv")

            if show_charts:
                st.subheader("📊 Insights")
                fig1, ax1 = plt.subplots()
                sns.countplot(data=df, x="Sentiment", ax=ax1, palette="viridis")
                ax1.set_title("Sentiment Distribution")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                sns.countplot(data=df, x="Tone", ax=ax2, palette="coolwarm")
                ax2.set_title("Tone Breakdown")
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
