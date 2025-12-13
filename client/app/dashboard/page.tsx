"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Calendar, MessageSquare, Clock, User, RotateCcw } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { Hamburger, MobileMenu } from "@/components/ui/hamburger";
import chatService, { ApiConversation } from "@/services/apiService";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import axios from "axios";
import { server } from "@/utils";
import { toast } from "sonner"; // Import toast from sonner

export interface Conversation {
  conversation_id: string;
  date: string;
  time: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const menuRef = useRef<HTMLDivElement>(null);
  const { fetchEmployeeProfile, logout, employeeData, check_role } = useAuth();
  const [pastConversations, setPastConversations] = useState<Conversation[]>(
    []
  );
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [scrolled, setScrolled] = useState(false);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    const checkAuth = async () => {
      // Check if token exists
      const token = localStorage.getItem("access_token");
      if (!token) {
        router.push("/"); // Redirect to login page if no token
        return;
      }

      try {
        // Verify token by loading the user profile
        const profile = await fetchEmployeeProfile();
        if (!profile || profile.role !== "employee") {
          router.push("/"); // Redirect if invalid profile or wrong role
        } else {
          fetchEmployeeConversations();
        }
      } catch (error) {
        console.error("Authentication error:", error);
        toast.error("Authentication failed. Redirecting to login.", {
          id: "auth-failed-redirect",
        });
        router.push("/"); // Redirect on any auth error
      }
    };

    checkAuth();
  }, [router]);

  const fetchEmployeeConversations = async () => {
    setIsLoading(true);
    try {
      const response = await chatService.getEmployeeConversations();

      if (response) {
        setPastConversations(response);
        toast.success("Conversations loaded successfully.", {
          id: "conversations-loaded",
        });
      } else {
        // If no conversations returned, set empty array
        setPastConversations([]);
        toast.info("No conversations found.", {
          id: "no-conversations-found",
        });
      }
    } catch (error) {
      console.error("Error fetching conversations:", error);
      toast.error("Failed to load conversations.", {
        id: "conversation-load-error",
      });
      setPastConversations([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConversationClick = (conversationId: string) => {
    router.push(`/dashboard/conversation/${conversationId}`);
  };

  const handleResetConversationStatus = async () => {
    if (!employeeData?.employee_id) {
      toast.error("Missing employee id.");
      return;
    }

    try {
      setResetting(true);
      const token = localStorage.getItem("access_token") || "";
      const response = await axios.put(
        `${server}/api/data/employee/${employeeData.employee_id}/change_conversation_status`,
        null,
        {
          params: { conversation_completed: false },
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      toast.success("Conversation status reset.", {
        description: response.data?.message,
        id: "conv-reset",
      });
      await fetchEmployeeProfile();
    } catch (error) {
      console.error("Failed to reset conversation status:", error);
      toast.error("Reset failed.", {
        description: "Please try again.",
        id: "conv-reset-error",
      });
    } finally {
      setResetting(false);
    }
  };

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-background/50 text-foreground">
      <Button
        variant="outline"
        size="icon"
        className="fixed top-3 right-3 z-50 h-8 w-8 text-xs"
        onClick={handleResetConversationStatus}
        disabled={resetting}
        title="Reset conversation status"
      >
        <RotateCcw size={14} className={resetting ? "animate-spin" : ""} />
      </Button>
      {/* Fixed Header */}
      <header
        className={`fixed top-0 left-0 right-0 z-40 transition-all duration-300 ${
          scrolled
            ? "py-4 bg-background/80 backdrop-blur-md shadow-md"
            : "py-4 bg-transparent"
        }`}
      >
        <div className="container mx-auto px-4 md:px-6 flex items-center justify-between">
          <div className="flex items-center">
            <Avatar className="h-10 w-10 mr-3 border-2 border-primary/30">
              <AvatarFallback className="bg-primary/20 text-foreground">
                {employeeData?.employee_name?.charAt(0) || "U"}
              </AvatarFallback>
            </Avatar>
            <div className="hidden sm:block">
              <h2 className="font-medium text-foreground">
                {employeeData?.employee_name || "Welcome"}
              </h2>
              <p className="text-xs text-foreground/70">
                {employeeData?.employee_id || "Employee"}
              </p>
            </div>
          </div>

          {/* <div className="text-center flex-1 sm:flex-none">
            <h1
              className={`font-bold text-gradient transition-all duration-300 ${
                scrolled ? "text-2xl" : "text-2xl md:text-3xl"
              }`}
            >
              Employee Dashboard
            </h1>
          </div> */}

          <div className="w-10 h-10">
            {/* Placeholder to balance the header */}
          </div>
        </div>
      </header>

      {/* Hamburger button stays fixed regardless of scroll */}
      <Hamburger
        isOpen={isMenuOpen}
        onClick={() => setIsMenuOpen(!isMenuOpen)}
      />

      <MobileMenu
        isOpen={isMenuOpen}
        onClose={() => setIsMenuOpen(false)}
        user={employeeData}
        logout={logout}
      />

      {/* Main content with proper padding to account for fixed header */}
      <main className="px-4 md:px-6 pb-6">
        <div className="w-full mx-auto">
          {/* Welcome Card (replaces original header) */}
          <div className=" p-6 rounded-xl mx-auto mb-10">
            <h2 className="text-xl md:text-2xl font-semibold text-primary-light mb-2 text-center">
              Welcome Back, {employeeData?.employee_name || "Employee"}!
            </h2>
            <p className="text-foreground/80 text-center">
              We're glad to see you here. Explore your dashboard to stay updated
              and connected.
            </p>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
            <div className="stats-card bg-[#004B23] text-white">
              <div className="flex items-center gap-4">
                <Calendar className="stats-icon" />
                <div>
                  <h3 className="text-white/80 text-sm font-medium">
                    Next Check-in
                  </h3>
                  <p className="text-white text-xl font-semibold">
                    Not scheduled yet
                  </p>
                </div>
              </div>
            </div>

            <div className="stats-card bg-[#004B23] text-white">
              <div className="flex items-center gap-4">
                <MessageSquare className="stats-icon" />
                <div>
                  <h3 className="text-white/80 text-sm font-medium">
                    Total Conversations
                  </h3>
                  <p className="text-white text-xl font-semibold">
                    {pastConversations.length}
                  </p>
                </div>
              </div>
            </div>

            <div className="stats-card bg-[#004B23] text-white">
              <div className="flex items-center gap-4">
                <Clock className="stats-icon" />
                <div>
                  <h3 className="text-white/80 text-sm font-medium">
                    Last Activity
                  </h3>
                  <p className="text-white text-xl font-semibold">
                    {pastConversations.length > 0
                      ? pastConversations[0].time
                      : "No recent activity"}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Past Conversations - keep your existing code */}
          <div className="bg-surface/40 p-6 rounded-xl">
            <h2 className="text-xl font-semibold text-primary-light mb-4">
              Past Conversations
            </h2>
            <p className="text-foreground/80 mb-6">
              Review your previous well-being check-ins
            </p>

            <ScrollArea className="h-[350px] pr-4">
              {isLoading ? (
                <div className="flex items-center justify-center h-40">
                  <p className="text-foreground/80">Loading conversations...</p>
                </div>
              ) : pastConversations.length > 0 ? (
                <div className="space-y-4">
                  {pastConversations.map((conversation) => (
                    <div
                      key={conversation.conversation_id}
                      className="stats-card p-4 hover:bg-surface/50 rounded-lg transition-colors ease-in-out duration-200 cursor-pointer border border-border/30"
                      onClick={() =>
                        handleConversationClick(conversation.conversation_id)
                      }
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-medium text-foreground">
                            {conversation.date}
                          </h3>
                        </div>
                        <span className="text-xs text-foreground/50">
                          {conversation.time}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-40">
                  <p className="text-foreground/70">
                    No past conversations found
                  </p>
                </div>
              )}
            </ScrollArea>
          </div>
        </div>
      </main>
    </div>
  );
}
