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
   DATABASE_URL=postgresql://<username>:<password>@localhost:5432/<database_name>
   OPENAI_API_KEY=<your_openai_api_key>
   AWS_ACCESS_KEY_ID=<your_aws_access_key>
   AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
   ```

5. **Run Database Migrations**:
   ```bash
   alembic upgrade head
   ```

6. **Start the Backend Server**:
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
   NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
   ```

4. **Start the Frontend Server**:
   ```bash
   npm run dev
   ```
   The frontend will be available at: `http://localhost:3000`

---

### Backend Setup
1. **Navigate to the Backend Directory**:
    ```bash
    cd server
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**:
    ```bash
    venv/bin/activate  # Use `venv\Scripts\activate` on Windows
    ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Start the Backend Server**:
    ```bash
    uvicorn main:app --reload
    ```
    The backend will be available at: `http://127.0.0.1:8000`

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

### Frontend (Vercel)
- Push the `client` directory to a GitHub repository.
- Connect the repo to [Vercel](https://vercel.com/) and deploy.

### Backend (Render)
- Push the `Server` directory to a GitHub repository.
- Connect the repo to [Render](https://render.com/) and deploy.
