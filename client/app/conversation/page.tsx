"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Mic, MicOff, Send, Volume2, VolumeX } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import axios from "axios";
import { useAuth } from "@/context/AuthContext";
import Message from "@/components/Conversation/message";
import { server } from "@/utils";
import { set } from "zod";
import { toast } from "sonner"; // Import toast from sonner instead

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: string;
  msg_type: string; // normal_question, followup_1, followup_2, insights, welcome, user_msg
}

interface TypingIndicator {
  isActive: boolean;
}

export default function ConversationPage() {
  // const server = "http://127.0.0.1:8000";

  const TotalQuestions = 5;

  const { employeeData, check_role } = useAuth();
  const router = useRouter();
  const [employee_name, setEmployee_Name] = useState("");
  const [employee_id, setEmployee_Id] = useState("");
  const [shap, setShap] = useState<string[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(
    typeof window !== "undefined"
      ? localStorage.getItem("conversation_id")
      : null
  );
  const [selectedQuestions, setSelectedQuestions] = useState<string[]>([]);

  const [maxQuestions, setMaxQuestions] = useState(TotalQuestions);

  const [messages, setMessages] = useState<Message[]>([]);
  const [chatHistory, setChatHistory] = useState<
    { sender_type: string; message: string }[]
  >([]);

  const [inputValue, setInputValue] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [micPermission, setMicPermission] = useState<
    "granted" | "denied" | "pending" | "unknown"
  >("unknown");
  const [isSpeakerOn, setIsSpeakerOn] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState<TypingIndicator>({
    isActive: false,
  });
  const scrollRef = useRef<HTMLDivElement>(null);
  const recordingAnimationRef = useRef<NodeJS.Timeout | null>(null);
  const [recordingLevel, setRecordingLevel] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const [isAudioProcessing, setIsAudioProcessing] = useState(false);
  const [hasEnded, setHasEnded] = useState(false);
  const token =
    typeof window !== "undefined" && localStorage.getItem("access_token");
  const [message_type, setMessageType] = useState<string>("welcome");
  const [insightLoadingId, setInsightLoadingId] = useState<string | null>(null);

  useEffect(() => {
    try {
      if (!check_role("employee")) {
        localStorage.removeItem("access_token");
        router.push("/");
        return;
      }
    } catch (error) {
      console.log(error);
    }
  }, []);

  const checkForIncompleteConv = async () => {
    try {
      // console.log(employeeData?.employee_id);

      const conv_id = localStorage.getItem("conversation_id");

      // if (lastConversation.date === new Date().toISOString().split("T")[0]) {
      const messagesResponse = await axios.get(
        `${server}/api/conversation/history/${conv_id}`,
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Transform the message format
      let questionsAsked = 0;
      const formattedMessages: Message[] = messagesResponse.data.map(
        (msg: any) => {
          if (msg.sender_type === "chatbot") {
            if (msg.message_type === "normal_question") {
              questionsAsked++;
            } // Count chatbot messages
            setMessageType(msg.message_type);
          }
          console.log(msg.time);

          return {
            id: msg.id.toString(),
            content: msg.content,
            isUser: msg.sender_type === "employee", // isUser is true if the sender is an employee
            timestamp: msg.time,
            msg_type: msg.message_type,
          };
        }
      );
      console.log("questions Asked:", questionsAsked);
      // setMaxQuestions(TotalQuestions - questionsAsked - 1);
      console.log(maxQuestions);
      setMessages(formattedMessages);

      // Add just one toast notification with unique ID
      toast.success("Conversation Resumed", {
        description: "You're continuing your previous conversation.",
        id: "conversation-resumed", // Add unique ID to prevent duplicate toasts
      });

      // return true;
      // } else return false;
    } catch (error) {
      console.error("Error fetching conversations:", error);
      toast.error("Error", {
        description: "Failed to check for incomplete conversations",
        id: "failed-fetch-conversations", // Add unique ID for error toast
      });
      return false;
    }
  };

  const startConv = async () => {
    console.log("Starting Conversation");
    try {
      // Show typing indicator while waiting for the first message
      setIsTyping({ isActive: true });

      const response = await axios.get(`${server}/api/conversation/start`, {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });
      const data = response.data;
      // console.log("On Conv Page",data);
      localStorage.setItem("conversation_id", data.conversation_id);

      setConversationId(data.conversation_id);
      // setSelectedQuestions(data.selected_questions);

      // Hide typing indicator after receiving the response
      setIsTyping({ isActive: false });

      setMessages([
        {
          id: "welcome",
          content: data.chatbot_response,
          isUser: false,
          timestamp: new Date().toTimeString().split(" ")[0],
          msg_type: "welcome",
        },
      ]);
      return;
    } catch (error) {
      // Hide typing indicator in case of error
      setIsTyping({ isActive: false });

      console.error("Error starting conversation:", error);
      toast.error("Error", {
        description: "Failed to start the conversation. Please try again.",
        id: "conversation-start-failed",
      });
    }
  };

  const handleCheckForIncompleteConv = async () => {
    // const res = await checkForIncompleteConv();  // Wait for the promise to resolve
    // console.log("CheckIncompleteConv: ", res);  // Log the result of checkForIncompleteConv

    // if (!res) {
    //   startConv();  // Only call startConv if res is falsy (or whatever condition you have)
    // }
    console.log("In handleCheckForIncompleteConv")
    const conv_id =
      typeof window !== "undefined"
        ? localStorage.getItem("conversation_id")
        : null;
    if (conv_id) {
      await checkForIncompleteConv();
    } else {
      console.log("No conversation ID found, starting new conversation.");
      await startConv();
      await handleSendMessage();
    }
  };

  useEffect(() => {
    if (employeeData) {
      console.log("Setting employee data from auth context:", employeeData);
      setEmployee_Id(employeeData.employee_id);
      setEmployee_Name(employeeData.employee_name);
      // setShap(employeeData.shap_values);
      // handleCheckForIncompleteConv();
      startConv();
    }
  }, [employeeData]);

  // useEffect(() => {
  //   handleCheckForIncompleteConv();
  // }, [employeeData]);

  // setConversationId(data.conversation_id);

  // setSelectedQuestions(data.selected_questions);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    if (isRecording) {
      recordingAnimationRef.current = setInterval(() => {
        setRecordingLevel(Math.random() * 0.8 + 0.2);
      }, 150);
    } else {
      if (recordingAnimationRef.current) {
        clearInterval(recordingAnimationRef.current);
      }
      setRecordingLevel(0);
    }

    return () => {
      if (recordingAnimationRef.current) {
        clearInterval(recordingAnimationRef.current);
      }
    };
  }, [isRecording]);

  useEffect(() => {
    if (navigator.mediaDevices) {
      navigator.permissions
        .query({ name: "microphone" as PermissionName })
        .then((permissionStatus) => {
          setMicPermission(
            permissionStatus.state === "prompt"
              ? "pending"
              : (permissionStatus.state as "granted" | "denied")
          );

          permissionStatus.onchange = () => {
            setMicPermission(
              permissionStatus.state === "prompt"
                ? "pending"
                : (permissionStatus.state as "granted" | "denied")
            );
          };
        })
        .catch(() => {
          // Fallback for browsers that don't support permissions API
          setMicPermission("unknown");
        });
    } else {
      setMicPermission("denied");
    }
  }, []);

  const provideInsights = async () => {
    try {
      const loadingId = `insight-loading-${Date.now()}`;
      setInsightLoadingId(loadingId);
      setMessages((prev) => [
        ...prev,
        {
          id: loadingId,
          content: "Generating insights based on your conversation...",
          isUser: false,
          timestamp: new Date().toTimeString().split(" ")[0],
          msg_type: "insight_loading",
        },
      ]);

      const response = await axios.get(
        `${server}/api/conversation/insights/${conversationId}`,
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const data = await response.data;
      console.log(data);
      const botMessage: Message = {
        id: `bot-${Date.now()}`,
        content: data.insights,
        isUser: false,
        timestamp: new Date().toTimeString().split(" ")[0],
        msg_type: "insights",
      };

      setMessages((prev) =>
        prev.map((msg) => (msg.id === loadingId ? botMessage : msg))
      );
      setInsightLoadingId(null);
    } catch (error) {
      console.error("Error providing insights:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === insightLoadingId
            ? {
                ...msg,
                content:
                  "We couldn't generate insights right now. Please try again.",
                msg_type: "insight_error",
              }
            : msg
        )
      );
      toast.error("Error", {
        description: "Failed to generate insights. Please try again later.",
        id: "insights-error",
      });
      setInsightLoadingId(null);
    }
  };

  const generateReport = async () => {
    try {
      // console.log(shapValues);

      const response = await axios.post(
        `${server}/api/report/employee`,
        {
          conversation_id: conversationId,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.data;
      console.log(data);
    } catch (error) {
      console.error("Error generating report:", error);
      toast.error("Error", {
        description:
          "Failed to generate your wellbeing report. Please try again later.",
        id: "error-type",
      });
    }
  };

  const handleSendMessage = async (overrideText?: string) => {
    // Do not send after the flow has ended
    if (hasEnded) return;

    setIsLoading(true);

    const textToSend = (overrideText ?? inputValue).trim();

    const chat_history = messages.map((msg) => ({
      sender_type: msg.isUser ? "employee" : "chatbot",
      message: msg.content,
    }));

    if (textToSend) {
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        content: textToSend,
        isUser: true,
        timestamp: new Date().toTimeString().split(" ")[0],
        msg_type: "user_msg",
      };
      setMessages((prev) => [...prev, userMessage]);
      setInputValue("");
    }

    setIsTyping({ isActive: true });

    try {
      // Determine what the user just answered (based on previous bot message type)
      const answeredAQuestion =
        message_type === "normal_question" ||
        message_type === "followup_1" ||
        message_type === "followup_2";
      const remainingAfterAnswer = answeredAQuestion
        ? Math.max(maxQuestions - 1, 0)
        : maxQuestions;

      const response = await axios.post(
        `${server}/api/conversation/message`,
        {
          message: textToSend || "",
          conversation_id:
            conversationId || localStorage.getItem("conversation_id"),
          message_type: message_type,
          chat_history: chat_history,
          question_set: selectedQuestions,
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const data = await response.data;

      const botMessage: Message = {
        id: `bot-${Date.now()}`,
        content: data.chatbot_response,
        msg_type: data.message_type,
        isUser: false,
        timestamp: new Date().toTimeString().split(" ")[0],
      };

      setIsTyping({ isActive: false });
      setSelectedQuestions(data.question_set);
      setMessageType(data.message_type);
      setMaxQuestions(remainingAfterAnswer);

      if (answeredAQuestion && remainingAfterAnswer <= 0) {
        // Store the final answer server-side, but don't surface the next question
        setHasEnded(true);
        await provideInsights();
        await generateReport();
      } else {
        setMessages((prev) => [...prev, botMessage]);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error("Connection Error", {
        description:
          "Failed to send your message. Please check your connection and try again.",
        id: "send-message-connection-error",
      });
      setIsTyping({ isActive: false });
      setIsLoading(false);
      setMessages((prev) => prev.slice(0, prev.length - 1));
      return;
    }

    setIsLoading(false);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start();
      console.log("Recording started");

      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
      alert(
        "Could not start recording. Please check your microphone permissions."
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      setIsRecording(false);
      console.log("Stopping recording...");
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        await processAudio(audioBlob);

        // Release the media recorder
        mediaRecorderRef.current = null;
        audioChunksRef.current = [];
      };
      mediaRecorderRef.current.stop();
      console.log("Recording stopped");
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsLoading(true);
    setIsAudioProcessing(true);
    try {
      // Convert speech to text
      const formData = new FormData();
      formData.append("audio", audioBlob);

      const transcriptionResponse = await axios.post(
        `${server}/api/conversation/transcribe`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      const transcribedText = transcriptionResponse.data.transcript;
      console.log("Transcribed text", transcribedText);

      setInputValue(transcribedText);
      // Immediately send the transcribed text without waiting for manual submit
      await handleSendMessage(transcribedText);
    } catch (error: any) {
      console.error("Error processing audio:", error);
      const detail =
        (error?.response?.data?.detail as string) ||
        (error?.message as string) ||
        "Failed to transcribe your voice. Please try typing instead.";
      toast.error("Transcription unavailable", {
        description: detail,
        id: "transcription-error",
      });
    } finally {
      setIsAudioProcessing(false);
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !isLoading) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  useEffect(() => {
    return () => {
      if (audioPlayerRef.current) {
        URL.revokeObjectURL(audioPlayerRef.current.src);
      }
    };
  }, []);

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <ScrollArea className="flex-1 p-4">
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="text-center py-6">
            <h1 className="text-2xl md:text-3xl font-bold mb-2 text-gradient">
                Well-being Check-in
            </h1>
            <p className="text-muted-foreground">
              Your responses are confidential and will help us provide better
              support.
            </p>
          </div>

          {messages.map((message) => (
            <Message
              key={message.id}
              id={message.id}
              content={message.content}
              isUser={message.isUser}
              time={message.timestamp}
              msg_type={message.msg_type}
            />
          ))}

          {/* Typing indicator */}
          {isTyping.isActive && (
            <div className="flex justify-start mb-4">
              <div className="flex items-start max-w-[80%]">
                <Avatar className="h-10 w-10 mr-3">
                  <AvatarImage src="/favicon.ico" alt="Bot Avatar" />
                  <AvatarFallback>DC</AvatarFallback>
                </Avatar>
                <div className="relative">
                  <div className="px-4 py-3 rounded-lg bg-secondary text-foreground rounded-bl-none flex items-center">
                    <div className="flex space-x-1">
                      <div
                        className="h-2 w-2 rounded-full animate-bounce"
                        style={{
                          animationDelay: "0ms",
                          backgroundColor: "#86BC25", //   green hex color
                        }}
                      ></div>
                      <div
                        className="h-2 w-2 rounded-full animate-bounce"
                        style={{
                          animationDelay: "150ms",
                          backgroundColor: "#86BC25", //   green hex color
                        }}
                      ></div>
                      <div
                        className="h-2 w-2 rounded-full animate-bounce"
                        style={{
                          animationDelay: "300ms",
                          backgroundColor: "#86BC25", //   green hex color
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      <div className="p-4">
        {hasEnded && (
          <div className=" fixed bottom-4 right-4 z-10">
            <Button
              className="bg-black text-white hover:bg-gray-800"
              onClick={() => {
                localStorage.removeItem("conversation_id"),
                  router.push("/dashboard");
              }}
            >
              Back to Dashboard
            </Button>
          </div>
        )}
        <div className="max-w-4xl mx-auto flex items-end gap-2">
          <div className="flex gap-4 relative flex-row w-full items-center bg-[#212835] p-3 rounded-xl">
            <div className="bg-secondary flex flex-row w-full items-center px-2 rounded-xl">
              <Textarea
                placeholder="Type your message..."
                className="flex-1  min-h-[60px]  resize-none max-h-[200px] bg-secondary text-foreground transition-all duration-300  break-words  "
                style={{
                  // border: inputValue.trim()
                  //   ? "2px solid #86BC25"
                  //   : "1px solid transparent",
                  borderRadius: "0.375rem",
                  overflowWrap: "break-word",
                  wordWrap: "break-word",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}
                value={inputValue}
                wrap="hard"
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading || hasEnded || isRecording}
              />
              {/* Microphone button with improved states */}
              <div className="h-[60px] self-end flex items-center">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={isRecording ? stopRecording : startRecording}
                  style={{
                    backgroundColor:
                      isRecording || isAudioProcessing ? "#86BC25" : "",
                    color: isRecording || isAudioProcessing ? "black" : "",
                    opacity: micPermission === "denied" ? 0.5 : 1,
                    cursor:
                      micPermission === "denied" ? "not-allowed" : "pointer",
                  }}
                  disabled={micPermission === "denied" || isAudioProcessing}
                  className="relative overflow-hidden shrink-0"
                  title={
                    micPermission === "denied"
                      ? "Microphone access denied"
                      : isRecording
                      ? "Stop recording"
                      : isAudioProcessing
                      ? "Processing audio..."
                      : "Start recording"
                  }
                >
                  {/* onClick={isRecording ? stopRecording : startRecording}
            className={`${
              isRecording ? "bg-destructive text-destructive-foreground" : ""
              }`}
              disabled={isLoading} */}
                  {/* Different states for microphone button */}
                  {micPermission === "pending" ? (
                    <div className="animate-pulse">
                      <Mic className="h-5 w-5" />
                    </div>
                  ) : isRecording ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="flex items-end space-x-1">
                        <div
                          className="w-1 rounded-t-full transition-all duration-150"
                          style={{
                            height: `${recordingLevel * 16}px`,
                            backgroundColor: "black",
                          }}
                        ></div>
                        <div
                          className="w-1 rounded-t-full transition-all duration-150"
                          style={{
                            height: `${recordingLevel * 24}px`,
                            backgroundColor: "black",
                          }}
                        ></div>
                        <div
                          className="w-1 rounded-t-full transition-all duration-150"
                          style={{
                            height: `${recordingLevel * 20}px`,
                            backgroundColor: "black",
                          }}
                        ></div>
                        <div
                          className="w-1 rounded-t-full transition-all duration-150"
                          style={{
                            height: `${recordingLevel * 18}px`,
                            backgroundColor: "black",
                          }}
                        ></div>
                      </div>
                    </div>
                  ) : isAudioProcessing ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="h-5 w-5 border-2 border-t-transparent border-black rounded-full animate-spin"></div>
                    </div>
                  ) : (
                    <Mic className="h-5 w-5" />
                  )}
                </Button>
              </div>
            </div>
            {/* Send button */}

            <div className="h-[60px] self-end flex items-center">
              <Button
                style={{
                  backgroundColor:
                    inputValue.trim() && !isLoading ? "green-300" : "green-500",
                  // opacity: inputValue.trim() && !isLoading ? 1 : 1,
                }}
                className="bg-green-500 hover:bg-green-600 text-black cursor-pointer shrink-0"
                size="icon"
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
