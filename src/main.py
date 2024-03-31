# Import needed libraries
import os
import logging
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import pandas as pd
from dotenv import load_dotenv
from questions import answer_question
import openai

# Loading environment varialbes
load_dotenv()
# Get API keys from .env
openai.api_key = os.environ["OPENAI_API_KEY"]
tg_bot_token = os.environ['TG_BOT_TOKEN']

# Pull in embeddings
df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

# How we interact with chat system (Memory) array of objects, content is text of all the questions
# messages is what we send to openai api
messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}]

# Logging to get feedback
logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO)

# start command - async chat function
# define for each different sections, define a function, function takes update and context,
# Update has all the chat information
# context is the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")
  
# Update is all the messages in the telegram chat
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
  # update.message.txt is what the user just sent to the chatbot
  messages.append({"role": "user", "content": update.message.text})
  # Calling out to openai - takes two arguments (model and message)
  completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
  # Openai gives back a large "completion" object, need to get just the answer, the LLM response
  completion_answer = completion['choices'][0]['message']['content']
  # We want to append the answer to the message array (message is the memory of the BOT)
  messages.append({"role": "assitant", "content": completion_answer})
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=completion_answer)
  
async def question(update: Update, context:ContextTypes.DEFAULT_TYPE):
  # update is an object that as all the chat history from telegram. We can take the message.text of the chat history to get the question asked
  answer = answer_question(df, question=update.message.text)
  await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()
  
  # Anytime the user types "/start", program will run the start async def function above 
  start_handler = CommandHandler('start', start)
  # filters is a way for the telegram api to filter types of media sent to telegram
  chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)
  # Command handler means that it is a command in a chat (use /command) to invoke
  question_handler = CommandHandler('question', question)


  application.add_handler(start_handler)
  application.add_handler(chat_handler)
  application.add_handler(question_handler)

  application.run_polling()