# HR-Employee Chatbot System

## Project Description

The **HR-Employee Chatbot System** is an AI-powered solution designed to enhance employee engagement and well-being tracking. It integrates multiple datasets (e.g., Vibemeter, performance reviews, leave records) to identify employees requiring engagement, conduct personalized conversations, and provide actionable insights for HR teams. The system automates HR workflows, reduces manual effort, and fosters a positive workplace culture.

### Key Features
- **SHAP Integration**: Identifies employees requiring engagement based on SHAP values.
- **Personalized Conversations**: Context-aware chatbot interactions powered by GPT-4.
- **Sentiment Analysis**: Real-time emotion detection using DistilBERT.
- **HR Dashboard**: Displays analytics, flagged employees, and detailed reports.
- **Automated Reporting**: Generates daily and employee-specific reports.

---

## Tech Stack

### Frontend
- **Framework**: Next.js  
- **Styling**: TailwindCSS  
- **Deployment**: Vercel  

### Backend
- **Framework**: FastAPI  
- **Database**: PostgreSQL  
- **AI Models**: GPT-4, DistilBERT  
- **Deployment**: Render  

---

## Steps to Run the Project

### Prerequisites
1. **Node.js** (v16+): [Download](https://nodejs.org/)  
2. **Python** (v3.9+): [Download](https://www.python.org/)  
3. **PostgreSQL**: Install and configure a PostgreSQL database  
4. **Environment Variables**: Create `.env` files for both frontend and backend

---

### Backend Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/hr-employee-chatbot.git
   cd hr-employee-chatbot/Server
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:  
   Create a `.env` file in the `Server` directory:
   ```env
   See the example env inside the `Server` directory 
   ```

5. **Start the Backend Server**:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be available at: `http://127.0.0.1:8000`

---

### Frontend Setup

1. **Navigate to the Frontend Directory**:
   ```bash
   cd ../client
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Set Up Environment Variables**:  
   Create a `.env.local` file:
   ```env
   See the example env inside the `client` directory
   ```

4. **Start the Frontend Server**:
   ```bash
   npm run dev
   ```
   The frontend will be available at: `http://localhost:3000`

---
## Testing the System

1. **Access the Frontend**:  
   Open `http://localhost:3000` in your browser.

2. **API Documentation**:  
   Visit `http://127.0.0.1:8000/docs` (Swagger UI) for testing APIs.

3. **Database**:  
   Use a tool like pgAdmin to manage and inspect the PostgreSQL database.

---

## Deployment
