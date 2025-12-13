"use client";

import React, { useState, useEffect } from "react";
import {
  BarChart,
  FileText,
  LogOut,
  User,
  Menu,
  LucideIcon,
  Briefcase,
  Mail,
  Phone,
  Upload,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/context/AuthContext";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  className?: string;
  hrName?: string;
  hrEmail?: string;
  hrId?: string;
}

export function Sidebar({
  collapsed,
  onToggle,
  hrName,
  hrEmail,
  hrId,
}: SidebarProps) {
  const [activeSection, setActiveSection] = useState<string>("analytics-section");
  // Track if sidebar is completely hidden (mobile/tablet views)
  const [isHidden, setIsHidden] = useState(false);
  // Track screen size for responsive behavior
  const [isMobile, setIsMobile] = useState(false);
  const initials =
    hrName
      ?.split(" ")
      .map((part) => part.charAt(0).toUpperCase())
      .join("") || "HR";

  // Update isMobile state based on window width
  useEffect(() => {
    const checkScreenSize = () => {
      setIsMobile(window.innerWidth < 1024); // Consider mobile/tablet if less than 1024px
    };
    
    // Initial check
    checkScreenSize();
    
    // Add event listener for window resize
    window.addEventListener('resize', checkScreenSize);
    
    // Clean up on unmount
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);
  
  // When in mobile mode, we want the sidebar to be either fully visible or fully hidden
  // useEffect(() => {
  //   if (isMobile) {
  //     setIsHidden(collapsed);
  //   } else {
  //     setIsHidden(false);
  //   }
  // }, [collapsed, isMobile]);
  
  useEffect(() => {
  setIsHidden(false); // Never hide on mobile
}, [collapsed, isMobile]);
 

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
      setActiveSection(id);
    }
    
    // Auto-close sidebar on mobile after section selection
    if (isMobile) {
      onToggle();
    }
  };
  useEffect(() => {
    const sectionIds = ["analytics-section", "reports-section", "upload-data-section"];
  
    const observer = new IntersectionObserver(
      (entries) => {
        const visibleSection = entries.find((entry) => entry.isIntersecting);
        if (visibleSection?.target?.id) {
          setActiveSection(visibleSection.target.id);
        }
      },
      {
        root: null,
        rootMargin: "0px",
        threshold: 0.6, // 60% of section should be visible
      }
    );
  
    sectionIds.forEach((id) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });
  
    return () => observer.disconnect();
  }, []);
  const { logout } = useAuth();

  return (
    <>
      {/* Mobile overlay backdrop when sidebar is open */}
      {/* {isMobile && !isHidden && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )} */}
    
      {/* Sidebar - positioned fixed on mobile, and relative on desktop */}
      <div
        className={cn(
          "flex flex-col h-screen bg-hr-black border-r border-[#26890d]/30 shadow-sidebar transition-all duration-300 ease-in-out z-50",
          // Basic positioning - fixed for mobile, absolute for sliding overlay effect
          "fixed top-0 left-0 lg:absolute",
          // Width variations based on state
          isHidden ? "-translate-x-full" : 
            collapsed ? "w-[70px]" : "w-[240px]",
        )}
      >
        <div className="flex items-center justify-between p-4">
          {!collapsed && (
            <div className="font-semibold text-lg text-[#26890d] animate-fade-in">
              <span className="text-[#26890d]">Vibemeter HR</span>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="text-[#26890d] transition-colors duration-200"
            style={{
              ["--hover-bg-color" as string]: "rgba(38, 137, 13, 0.2)",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = "rgba(38, 137, 13, 0.2)";
              e.currentTarget.style.color = "white";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = "";
              e.currentTarget.style.color = "#26890d";
            }}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isMobile && !collapsed ? <X size={20} /> : <Menu size={20} />}
          </Button>
        </div>

        <Separator className="my-2 bg-[#26890d]/20" />

        <div className="flex-1 overflow-y-auto hr-scrollbar px-2 py-4 stagger-fade-in">
          <NavItem
            icon={BarChart}
            label="Analytics"
            active={activeSection === "analytics-section"}
            collapsed={collapsed}
            onClick={() => scrollToSection("analytics-section")}
          />
          <NavItem
            icon={FileText}
            label="Reports"
            active={activeSection === "reports-section"}
            collapsed={collapsed}
            onClick={() => scrollToSection("reports-section")}
          />
          <NavItem
            icon={Upload}
            label="Upload Data"
            active={activeSection === "upload-data-section"}
            collapsed={collapsed}
            onClick={() => scrollToSection("upload-data-section")}
          />
        </div>

        <Separator className="my-2 bg-[#26890d]/20" />

        <div className="p-4">
          <div
            className={cn(
              "flex flex-col items-center rounded-lg p-4 transition-all duration-300",
              collapsed
                ? "py-2 bg-transparent hover:bg-transparent"
                : "py-4 bg-[#26890d]/10 hover:bg-[#26890d]/20"
            )}
          >
            <Avatar
              className={cn(
                "h-20 w-20 border-2 border-[#26890d] shadow-lg mb-4",
                collapsed && "h-10 w-10 mb-2"
              )}
            >
              <AvatarImage src="https://github.com/shadcn.png" />
              <AvatarFallback className="bg-[#26890d] text-black font-semibold">
                {initials}
              </AvatarFallback>
            </Avatar>

            {!collapsed && (
              <div className="text-center space-y-3 animate-fade-in w-full">
                <div className="font-semibold text-[#26890d] text-lg">
                  {hrName || "HR Manager"}
                </div>
                <div className="text-sm text-gray-200 font-medium bg-[#26890d]/20 py-1 px-2 rounded-md">
                  HR Manager
                </div>

                <div className="space-y-2 pt-2 text-left">
                  <div className="flex items-center gap-2 text-xs text-gray-300">
                    <Briefcase size={14} className="text-[#26890d]" />
                    <span>ID: {hrId || "HR-2024-001"}</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-300">
                    <Mail size={14} className="text-[#26890d]" />
                    <span>{hrEmail || "hr@example.com"}</span>
                  </div>
                </div>

                <Button
                  variant="default"
                  onClick={logout}
                  className="w-full mt-2 justify-center gap-2 bg-[#252525] text-[#26890d] border-[#26890d] hover:bg-[#26890d] hover:text-black transition-colors duration-200"
                >
                  <LogOut size={16} />
                  <span>Log Out</span>
                </Button>
              </div>
            )}

            {collapsed && (
              <Button
                variant="default"
                className="p-2 mt-2 justify-center bg-[#252525] text-[#26890d] hover:bg-[#26890d] hover:text-black transition-colors duration-200"
                onClick={() => {
                  logout();
                  onToggle();
                }}
              >
                <LogOut size={16} />
              </Button>
            )}
          </div>
        </div>
      </div>
      
      {/* Mobile-only floating menu button to open sidebar when completely hidden */}
      {isMobile && isHidden && (
        <Button
          variant="default"
          size="icon"
          className="fixed left-4 top-4 z-40 bg-[#26890d] text-white hover:bg-[#26890d]/80 shadow-lg"
          onClick={onToggle}
          aria-label="Open menu"
        >
          <Menu size={20} />
        </Button>
      )}
    </>
  );
}

interface NavItemProps {
  icon: LucideIcon;
  label: string;
  active?: boolean;
  collapsed: boolean;
  onClick?: () => void;
}

function NavItem({
  icon: Icon,
  label,
  active,
  collapsed,
  onClick,
}: NavItemProps) {
  return (
    <Button
      variant="ghost"
      className={cn(
        "w-full justify-start gap-2 my-1 transition-all duration-200",
        collapsed ? "justify-center p-2" : "px-3 py-2",
        active
          ? "bg-[#26890d] text-black font-semibold pointer-events-none" // Active state: no hover effect
          : "text-white hover:!bg-[#26890d]/20 hover:!text-white"
      )}
      onClick={onClick}
    >
      <Icon size={20} />
      {!collapsed && <span>{label}</span>}
    </Button>
  );
}
