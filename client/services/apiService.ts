import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// Interfaces that match the server response data structures
export interface ApiConversation {
  conversation_id: string;
  // employee_id: string;
  // employee_name: string;
  // message_ids: number[];
  date: string;
  time: string;
  // report?: string;
}

export interface ApiMessage {
  id: number;
  content: string;
  sender_type: "employee" | "chatbot";
  time: string;
  message_type: string;
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatService = {
  // Get conversation history for an employee
  getEmployeeConversations: async (): Promise<ApiConversation[]> => {
    try {
      // Get token from localStorage or your auth context
      const token = localStorage.getItem('access_token') || '';
      
      const response = await api.get(`/api/conversation/history/employee`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching employee conversations:', error);
      throw error;
    }
  },

  // Get messages for a specific conversation
  getConversationMessages: async (conversationId: string): Promise<ApiMessage[]> => {
    try {
      const token = localStorage.getItem('access_token') || '';
      const response = await api.get(`/api/conversation/history/${conversationId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching conversation messages:', error);
      throw error;
    }
  }
};

export default chatService;
