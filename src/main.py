# Import needed libraries
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv

# Loading environment varialbes
load_dotenv()
tg_bot_token = os.environ['TG_BOT_TOKEN']

# How we interact with chat system (Memory) array of objects, content is text of all the questions
messages = [{
    "role": "system",
    "content": "You are a helpful assitant that answers questions.",
}]

# Logging to get feedback
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# start command
# define for each different sections, define a function, function takes update and context,
# Update has all the chat information
# context is the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="I'm a bot, please talk to me")
    
if __name__ == '__main__':
    application = ApplicationBuilder().token(tg_bot_token).build()

    # Anytime the user types "/start", program will run the start async def function above 
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    application.run_polling