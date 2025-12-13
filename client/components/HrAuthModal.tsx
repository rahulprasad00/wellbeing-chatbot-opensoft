"use client";

import React, { useState } from "react";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { AnimatedButton } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import { z } from "zod";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner"; // Import toast from sonner

// CSS classes to be added to globals.css if not already there
const cssClasses = `
.auth-input {
  background-color: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  transition: all 0.3s ease;
}

.auth-input:focus {
  background-color: rgba(255, 255, 255, 0.1) !important;
  border-color: rgba(255, 255, 255, 0.2) !important;
  box-shadow: 0 0 0 2px rgba(38, 137, 13, 0.2) !important;
}

.hr-dialog-content {
  border-radius: 0.75rem !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04) !important;
  background: linear-gradient(145deg, rgba(25, 25, 25, 0.95), rgba(10, 10, 10, 0.95)) !important;
  backdrop-filter: blur(10px) !important;
  overflow: hidden !important;
}

.error-message {
  animation: shake 0.5s cubic-bezier(0.36, 0.07, 0.19, 0.97) both;
}

@keyframes shake {
  10%, 90% {
    transform: translateX(-1px);
  }
  20%, 80% {
    transform: translateX(2px);
  }
  30%, 50%, 70% {
    transform: translateX(-4px);
  }
  40%, 60% {
    transform: translateX(4px);
  }
}
`;

export default function HRLoginModal() {
  const { signInHR } = useAuth();
  const router = useRouter();

  // Animation variants
  const formVariants = {
    hidden: {
      opacity: 0,
      y: 20,
      scale: 0.95,
    },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration: 0.4,
        ease: "easeOut",
      },
    },
    exit: {
      opacity: 0,
      y: -20,
      scale: 0.95,
      transition: {
        duration: 0.3,
        ease: "easeIn",
      },
    },
  };

  // zod validation
  const loginSchema = z.object({
    email: z.string().regex(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/).max(50),
    password: z.string().regex(
        /^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{6,}$/,
        "Password must contain at least 8 characters, including letters, numbers, and special characrs"
      ).max(20),
  });

  const [LoginformData, setLoginFormData] = React.useState({
    email: "",
    password: "",
  });

  // Modal state
  const [open, setOpen] = useState(false);

  // Login form states
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [Loginerrors, setLoginErrors] = React.useState({
    email: 0,
    password: 0
  });

  const handleLoginChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const updatedFormData = { ...LoginformData, [name]: value };
    setLoginFormData(updatedFormData);
    const fieldSchema = loginSchema.shape[name as keyof typeof loginSchema.shape];
    const result = fieldSchema.safeParse(value);

    setLoginErrors(prevErrors => ({
      ...prevErrors,
      [name]: result.success ? 0 : 1,
    }));
  };

  // Handle form submission
  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Form validation
    if (!LoginformData.email || !LoginformData.password) {
      setError("Please fill in all fields");
      toast.error("Login Failed", {
        description: "Please fill in all fields",
        id: "hr-login-form-error",
      });
      return;
    }

    // Show loading toast
    toast.loading("Logging in to HR dashboard...", { id: "hr-login-attempt" });
    setLoading(true);

    try {
      // Call the HR login function
      const response = await signInHR(LoginformData.email, LoginformData.password);
      if (response.token) {
        // Dismiss loading toast
        toast.dismiss("hr-login-attempt");
        
        // Close the modal after successful login
        setOpen(false);
        
        // Redirect to HR Dashboard on success
        router.push("/hr-dashboard");
      }
    } catch (err: any) {
      // Dismiss loading toast and show error
      toast.dismiss("hr-login-attempt");
      setError("Invalid email or password, or you are not authorized as HR.");
      toast.error("HR Login Failed", {
        description: "Invalid credentials or insufficient permissions",
        id: "hr-login-error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {/* Trigger button to open the modal */}
      <DialogTrigger asChild>
        <AnimatedButton
          className="bg-white border-1 border-black/50 text-black cursor-pointer text-lg font-semibold px-10 py-4 h-auto rounded-full shadow-md hover:shadow-lg hover:bg-white/95 hover:border-black/90 transition-all duration-300 group relative overflow-hidden md:scale-90 lg:scale-100 scale-75 "
        >
          Login as HR
        </AnimatedButton>
      </DialogTrigger>

      {/* Modal content */}
      <DialogContent className="sm:max-w-[425px] dark bg-[#131313] text-white border-none hr-dialog-content p-0 overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 to-teal-500"></div>

        <DialogHeader className="p-6 pb-2">
          <DialogTitle className="text-center text-2xl font-bold">
            HR Login
          </DialogTitle>
          <p className="text-center text-gray-400 text-sm mt-2">
            Access the MoodPulse HR dashboard
          </p>
        </DialogHeader>

        {/* Login form with animation */}
        <div className="p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key="hr-login-form"
              initial="hidden"
              animate="visible"
              exit="exit"
              variants={formVariants}
            >
              <form onSubmit={handleLoginSubmit} className="grid gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="email" className="text-sm font-medium">                    Email                  </Label>
                  <Input
                    id="email"
                    type="email"
                    name="email"
                    onChange={handleLoginChange}
                    placeholder="HR@example.com"
                    required={true}
                    className="auth-input p-3 rounded-md"
                  />
                  {Loginerrors.email ? (
                    <motion.p
                      initial={{ opacity: 0, y: -5 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-red-500 text-xs mt-1"
                    >
                      Enter valid email.
                    </motion.p>
                  ) : null}
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="password" className="text-sm font-medium">Password</Label>
                  <Input                    id="password"
                    type="password"
                    name="password"
                    onChange={handleLoginChange}
                    placeholder="••••••••"
                    required={true}
                    className="auth-input p-3 rounded-md"
                  />
                  {Loginerrors.password ? (
                    <motion.p
                      initial={{ opacity: 0, y: -5 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-red-500 text-xs mt-1"
                    >
                      Password must contain at least 8 characters, including
                      letters, numbers, and special characters.
                    </motion.p>
                  ) : null}
                </div>

                {/* Error message with animation */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-md text-sm error-message"
                  >
                    {error}
                  </motion.div>
                )}

                {/* Submit button */}
                <DialogFooter className="mt-4">
                  <AnimatedButton
                    type="submit"
                    disabled={loading}
                    className="cursor-pointer font-medium text-base w-full py-3"
                  >
                    {loading ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg
                          className="animate-spin h-5 w-5"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                        Logging In...
                      </span>
                    ) : (
                      "Sign In to HR Dashboard"
                    )}
                  </AnimatedButton>
                </DialogFooter>
              </form>
            </motion.div>
          </AnimatePresence>
        </div>
      </DialogContent>
    </Dialog>
  );
}