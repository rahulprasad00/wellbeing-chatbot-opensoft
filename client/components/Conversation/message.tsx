"use client";
import React, { useEffect } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { server } from "@/utils";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Volume2, VolumeX } from "lucide-react";
import ReactMarkdown from "react-markdown";

type MessageProps = {
  id: string;
  content: string;
  isUser: boolean;
  time: string;
  msg_type: string;
};

function Message({ id, content, isUser, time, msg_type = "" }: MessageProps) {
  const audioPlayerRef = React.useRef<HTMLAudioElement>(null);
  const [isSpeakerOn, setIsSpeakerOn] = React.useState(false);
  const [audioUrl, setAudioUrl] = React.useState<string | null>(null);
  const date = new Date();
  if (time) {
    const [hours, minutes] = time.split(":");
    date.setHours(parseInt(hours), parseInt(minutes));
  }
  const [isLoading, setIsLoading] = React.useState(!isUser && content === "");

  // Process content to remove extra quotes and braces
  const cleanContent = String(content || "")
    .replace(/^[{'"`]+/, "")
    .replace(/[}'"`]+$/, "");

  const SpeakAloud = async () => {
    setIsSpeakerOn(true);
    try {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl("");
      }
      console.log("Fetching audio for:", cleanContent);

      const response = await axios.post(
        `${server}/api/conversation/tts`,
        { prompt: cleanContent },
        {
          responseType: "blob",
        }
      );
      const url = URL.createObjectURL(response.data);
      console.log("Audio URL:", url);

      setAudioUrl(url);
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause();
        audioPlayerRef.current.src = url;
        audioPlayerRef.current.play();
      }
    } catch (error) {
      console.error("Error fetching audio:", error);
    } finally {
      setIsSpeakerOn(false);
    }
  };

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // Enhanced message rendering based on message type
  const renderMessageContent = () => {
    if (!isUser && (content === "" || isLoading)) {
      return (
        <div className="flex items-center">
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      );
    }

    // Apply special formatting for insights or specific message types
    if (msg_type === "insight_loading") {
      return (
        <div className="flex items-center gap-2 text-sm text-foreground">
          <div className="h-3 w-3 rounded-full border-2 border-[#86BC25] border-t-transparent animate-spin" />
          <span>Generating insights based on your conversation...</span>
        </div>
      );
    }

    if (msg_type === "insight_error") {
      return (
        <div className="text-sm text-red-400">
          {cleanContent || "We couldn't generate insights right now. Please try again."}
        </div>
      );
    }

    if (msg_type === "insight") {
      return (
        <div className="markdown-content">
          <ReactMarkdown>{cleanContent}</ReactMarkdown>
        </div>
      );
    }

    // For regular messages with possible formatting
    return (
      <div className="markdown-content text-sm">
        <ReactMarkdown>{cleanContent}</ReactMarkdown>
      </div>
    );
  };

  return (
    <div
      key={id}
      className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}
    >
      <div className="flex items-start max-w-[80%]">
        {!isUser && (
          <Avatar className="h-10 w-10 mr-3">
            <AvatarImage src="/favicon.ico" alt="Bot Avatar" />
            <AvatarFallback>DC</AvatarFallback>
          </Avatar>
        )}

        <div className="relative">
          <div
            className={`px-4 py-3 rounded-lg ${
              isUser
                ? "bg-[#26890d] text-white rounded-br-none"
                : "bg-secondary text-foreground rounded-bl-none"
            } `}
          >
            <div className="w-full flex flex-col break-word">
              {renderMessageContent()}
            </div>
            <div className="flex justify-between items-center mt-1">
              <div className="text-xs opacity-70 text-right">
                {date.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  hour12: true,
                })}
              </div>

              {!isUser && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={SpeakAloud}
                  className={`h-6 w-6 p-0 ml-2 rounded-full ${
                    isSpeakerOn
                      ? "text-gray-400"
                      : "text-gray-400 hover:text-white"
                  }`}
                  title={isSpeakerOn ? "Stop speaking" : "Speak message"}
                >
                  {!isSpeakerOn ? (
                    <Volume2 className="h-4 w-4" />
                  ) : (
                    <VolumeX className="h-4 w-4" />
                  )}
                </Button>
              )}
            </div>
          </div>
          <audio ref={audioPlayerRef} />
        </div>

        {isUser && (
          <Avatar className="h-10 w-10 ml-3 ">
            <AvatarFallback>ME</AvatarFallback>
          </Avatar>
        )}
      </div>
    </div>
  );
}

export default Message;
