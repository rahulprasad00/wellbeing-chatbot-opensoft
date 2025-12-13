"use client";

import React, { useState, useEffect, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { ArrowLeft, MessageSquare } from "lucide-react";
import chatService, { ApiMessage } from "@/services/apiService";
import Message from "@/components/Conversation/message";

export default function ConversationPage() {
  const { id } = useParams();
  const router = useRouter();
  const { fetchEmployeeProfile, employeeData } = useAuth();
  const [messages, setMessages] = useState<ApiMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Check if user is authenticated
    let isMounted = true;
    async function checkProfile() {
      try {
        const token = localStorage.getItem("access_token");
        if (!token) {
          router.push("/");
          return;
        }

        const profile = await fetchEmployeeProfile();
        if (!profile) {
          router.push("/");
          return;
        }

        if (isMounted) {
          fetchConversationMessages();
        }
      } catch (error) {
        console.error("Error fetching profile", error);
        if (isMounted) {
          router.push("/");
        }
      }
    }

    checkProfile();

    return () => {
      isMounted = false;
    };
  }, [id, router]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current && messages.length > 0) {
      const scrollableElement = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollableElement) {
        scrollableElement.scrollTop = scrollableElement.scrollHeight;
      }
    }
  }, [messages]);

  const fetchConversationMessages = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const messagesData = await chatService.getConversationMessages(
        id as string
      );

      // Sort messages by ID to ensure chronological order
      const sortedMessages = [...messagesData].sort((a, b) => a.id - b.id);
      setMessages(sortedMessages);
    } catch (error) {
      console.error("Error fetching messages:", error);
      setError("Failed to load conversation messages. Please try again later.");
      setMessages([]);
    } finally {
      setIsLoading(false);
    }
  };

  const goBack = () => {
    router.push("/dashboard");
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-6">
      <div className="w-full mx-auto">
        <div className="flex items-center mb-6">
          <Button
            variant="outline"
            onClick={goBack}
            className="p-2 mr-4 cursor-pointer hover:bg-foreground/10"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <h1 className="text-2xl md:text-3xl font-bold text-primary-light">
            Conversation Details
          </h1>
        </div>

        <div className="stats-card mb-8">
          <div className="flex items-center gap-4">
            <MessageSquare className="stats-icon" />
            <div>
              <h3 className="text-foreground/80 text-sm font-medium">
                Conversation Date
              </h3>
              <p className="text-foreground text-xl font-semibold">{id}</p>
            </div>
          </div>
        </div>

        <div className="glass-card p-6 rounded-xl">
          <h2 className="text-xl font-semibold text-primary-light mb-4">
            Messages
          </h2>

          <ScrollArea className="h-[600px] pr-4" ref={scrollAreaRef}>
            {isLoading ? (
              <div className="flex items-center justify-center h-40">
                <p className="text-foreground/70">Loading messages...</p>
              </div>
            ) : error ? (
              <div className="flex items-center justify-center h-40">
                <p className="text-red-400">{error}</p>
              </div>
            ) : messages.length > 0 ? (
              <div className="space-y-6 p-2">
                {messages.map((message) => (
                  <Message
                    key={message.id}
                    id={message.id.toString()}
                    content={message.content}
                    isUser={message.sender_type !== "chatbot"}
                    time={message.time}
                    msg_type={message.message_type}
                  />
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-40">
                <p className="text-foreground/70">
                  No messages found for this conversation
                </p>
              </div>
            )}
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
