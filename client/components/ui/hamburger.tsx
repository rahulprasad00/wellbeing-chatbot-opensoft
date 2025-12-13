"use client";
import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

import { Employee } from "@/context/AuthContext"; // Adjust import path as needed

interface HamburgerProps {
  isOpen: boolean;
  onClick: () => void;
}

interface MobileMenuProps {
  isOpen: boolean;
  onClose: () => void;
  user: Employee | null;
  logout: () => void;
}

export const Hamburger: React.FC<HamburgerProps> = ({ isOpen, onClick }) => {
  return (
    <button
      className="hamburger-button fixed z-50 top-4 right-6 w-10 h-10 flex flex-col justify-center items-center cursor-pointer"
      onClick={onClick}
      aria-label={isOpen ? "Close menu" : "Open menu"}
    >
      <div className="relative w-7 h-5">
        <span
          className={`absolute left-0 h-0.5 w-full bg-foreground rounded-full transition-all duration-300 ${
            isOpen ? "top-1/2 -translate-y-1/2 rotate-45" : "top-0"
          }`}
        />
        <span
          className={`absolute left-0 h-0.5 w-full bg-foreground rounded-full transition-all duration-300 ${
            isOpen ? "opacity-0" : "top-1/2 -translate-y-1/2"
          }`}
        />
        <span
          className={`absolute left-0 h-0.5 w-full bg-foreground rounded-full transition-all duration-300 ${
            isOpen ? "top-1/2 -translate-y-1/2 -rotate-45" : "top-full"
          }`}
        />
      </div>
    </button>
  );
};

export const MobileMenu: React.FC<MobileMenuProps> = ({
  isOpen,
  onClose,
  user,
  logout,
}) => {
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      // Check if clicked element is not the menu and not the hamburger button
      if (
        menuRef.current &&
        !menuRef.current.contains(event.target as Node) &&
        !(event.target as HTMLElement).closest(".hamburger-button")
      ) {
        onClose();
      }
    }

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose]);

  return (
    <div
      ref={menuRef}
      className={`fixed inset-y-0 right-0 w-72 bg-background border-l border-border/30 shadow-lg z-40 transition-all duration-300 ease-in-out ${
        isOpen ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <div className="h-full flex flex-col p-6 pt-20">
        {" "}
        {/* Added pt-20 for hamburger space */}
        {/* Profile Section */}
        <div className="flex items-center gap-4 pb-6 border-b border-border">
          <Avatar className="h-14 w-14 bg-foreground/40 backdrop-blur-lg">
            {/* <AvatarImage src={user?.avatar} /> */}
            <AvatarFallback>
              {user?.employee_name?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
          <div>
            <h3 className="text-lg font-medium text-foreground">
              {user?.employee_name || "User Name"}
            </h3>
            <p className="text-sm text-foreground/70">
              {user?.employee_email || "user@example.com"}
            </p>
          </div>
        </div>
        {/* Menu Buttons */}
        <nav className="flex-1 flex flex-col gap-2 py-8">
          <Button
            variant="ghost"
            className="w-full justify-start py-4 text-foreground hover:bg-primary/50"
          >
            Contact Us
          </Button>
        </nav>
        {/* Logout Button */}
        <div className="mt-auto pt-4 border-t border-border/30">
          <Button
            variant="outline"
            className="w-full py-4 bg-card border-red-500/30 text-red-500 hover:bg-red-500/10"
            onClick={() => {
              logout();
              onClose();
            }}
          >
            Log Out
          </Button>
        </div>
      </div>
    </div>
  );
};
