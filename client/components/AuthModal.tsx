/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import { z } from "zod";
import * as React from "react";
import axios from "axios";
import { SafeParseReturnType } from "zod";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button, AnimatedButton } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FcGoogle } from "react-icons/fc";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import { FaTwitter } from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner"; // Import toast from sonner
import { PassThrough } from "stream";

// Add this to your globals.css

// API call to check employee ID via backend.
async function checkEmployeeId(employeeId: string): Promise<boolean> {
  try {
    const response = await fetch(
      `http://127.0.0.1:8000/api/user/check/${employeeId}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    const data = await response.json();
    return data.exists;
  } catch (error) {
    console.error("Error checking employee ID:", error);
    throw error;
  }
}

export default function AuthModal() {
  const {
    signIn,
    signUp,
    signInWithGoogle,
    signInWithTwitter,
    fetchEmployeeProfile,
    setIsLogged,
    setSignInModalVisible,
    signInModalVisible,
  } = useAuth();
  const router = useRouter();

  // Mode can be "login" or "register".
  const [mode, setMode] = React.useState<"login" | "register">("login");
  // Registration step: "checkId" or "registerForm"
  const [registerStep, setRegisterStep] = React.useState<
    "checkId" | "registerForm"
  >("checkId");

  // Employee ID check state for registration.
  const [regEmployeeId, setRegEmployeeId] = React.useState("");
  const [error, setError] = React.useState("");
  // Loading and error states.
  const [loading, setLoading] = React.useState(false);

  // Animation states
  const [showLoginForm, setShowLoginForm] = React.useState(true);
  const [showVerifyForm, setShowVerifyForm] = React.useState(false);
  const [showRegisterForm, setShowRegisterForm] = React.useState(false);

  //Zod vaidation
  const loginSchema = z.object({
    email: z
      .string()
      .regex(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/)
      .max(50),
    password: z
      .string()
      .regex(
        /^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{6,}$/,
        "Password must contain at least 8 characters, including letters, numbers, and special characters"
      )
      .max(20),
  });
  const registerSchema = z
    .object({
      name: z.string().max(50).min(1),
      email: z
        .string()
        .regex(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/)
        .max(50),
      password: z
        .string()
        .regex(
          /^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{6,}$/,
          "Password must contain at least 8 characters, including letters, numbers, and special characters"
        )
        .max(20),
      confirmPassword: z.string().max(20),
    })
    .refine((data) => data.password === data.confirmPassword, {
      message: "Passwords do not match",
      path: ["confirmPassword"],
    });

  const [RegisterformData, setRegisterFormData] = React.useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [LoginformData, setLoginFormData] = React.useState({
    email: "",
    password: "",
  });
  const [Loginerrors, setLoginErrors] = React.useState({
    email: 0,
    password: 0,
  });
  const [Registererrors, setRegisterErrors] = React.useState({
    name: 0,
    email: 0,
    password: 0,
    confirmPassword: 0,
  });

  React.useEffect(() => {
    // Manage which form to show based on mode and registerStep
    if (mode === "login") {
      setShowLoginForm(true);
      setShowVerifyForm(false);
      setShowRegisterForm(false);
    } else if (mode === "register" && registerStep === "checkId") {
      setShowLoginForm(false);
      setShowVerifyForm(true);
      setShowRegisterForm(false);
    } else if (mode === "register" && registerStep === "registerForm") {
      setShowLoginForm(false);
      setShowVerifyForm(false);
      setShowRegisterForm(true);
    }
  }, [mode, registerStep]);

  const handleLoginChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const updatedFormData = { ...LoginformData, [name]: value };
    setLoginFormData(updatedFormData);
    const fieldSchema =
      loginSchema.shape[name as keyof typeof loginSchema.shape];
    const result = fieldSchema.safeParse(value);

    setLoginErrors((prevErrors) => ({
      ...prevErrors,
      [name]: result.success ? 0 : 1,
    }));
  };
  const handleRegisterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const updatedFormData = { ...RegisterformData, [name]: value };
    setRegisterFormData(updatedFormData);
    if (name === "confirmPassword") {
      const result = registerSchema.safeParse(updatedFormData);
      setRegisterErrors((prevErrors) => ({
        ...prevErrors,
        confirmPassword: result.success ? 0 : 1,
      }));
    } else {
      // Validate only the specific field
      const fieldSchema =
        registerSchema._def.schema.shape[
          name as keyof typeof registerSchema._def.schema.shape
        ];
      const result = fieldSchema.safeParse(value);
      setRegisterErrors((prevErrors) => ({
        ...prevErrors,
        [name]: result.success ? 0 : 1,
      }));
    }
  };

  // Reset states when modal closes.
  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      setError("");
      setMode("login");
      setRegisterStep("checkId");
    }
    setSignInModalVisible(isOpen);
  };

  // After successful login or registration, fetch the employee profile
  // and route the user based on whether they are flagged.
  const handlePostAuth = async () => {
    try {
      const profile = await fetchEmployeeProfile();
      if (profile.is_selected && !profile.conversation_completed) {
        router.push("/conversation");
      } else {
        router.push("/dashboard");
      }
    } catch (err) {
      console.log(err);
    }
  };

  // Handle login submission using Firebase.
  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = loginSchema.safeParse(LoginformData);
    console.log(LoginformData)
    if (result.success) {
      setLoading(true);
      try {
        // Show loading toast
        toast.loading("Logging in...", { id: "login-attempt" });

        // await signIn(LoginformData.email, LoginformData.password);
        const res = await axios.post(
          "http://127.0.0.1:8000/api/user/login",
          {
            email: LoginformData.email,
            password: LoginformData.password,
          },
          {
            headers: {
              "Content-Type": "application/json",
            },
          }
        );
        console.log(res);
        const token = res.data.token;
        localStorage.setItem("access_token", token);
        setIsLogged(true);

        // Dismiss loading toast and show success
        toast.dismiss("login-attempt");
        toast.success("Login Successful", {
          description: "Welcome back! Redirecting to your dashboard...",
        });

        await handlePostAuth();
        setSignInModalVisible(false);
      } catch (err: unknown) {
        setError("Invalid Credentials");

        // Dismiss loading toast and show error
        toast.dismiss("login-attempt");
        toast.error("Login Failed", {
          description: "Invalid email or password. Please try again.",
          id: "login-error",
        });
      } finally {
        setLoading(false);
      }
    } else {
      const newError = { email: 0, password: 0 };
      result.error.errors.forEach((err) => {
        if (err.path.includes("email")) newError.email = 1;
        if (err.path.includes("password")) newError.password = 1;
      });
    }
  };

  // Handle employee ID verification for registration.
  const handleVerifyEmployeeId = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const exists = await checkEmployeeId(regEmployeeId);
      console.log(exists);
      if (!exists) {
        setError("Employee ID not found.");
      } else {
    setRegisterStep("registerForm");
      }
    } catch {
      setError("Error checking employee ID. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Handle registration submission using Firebase.
  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    const result = registerSchema.safeParse(RegisterformData);
    console.log(result);
    if (result.success) {
      console.log("form submitted");
      try {
        await signUp(
          RegisterformData.email,
          RegisterformData.password,
          RegisterformData.name
        );
        const res = await axios.post(
          "http://127.0.0.1:8000/api/user/register",
          {
            email: RegisterformData.email,
            emp_id: regEmployeeId,
            name: RegisterformData.name,
            password: RegisterformData.password
          }
        );
        const token = res.data.token;
        localStorage.setItem("access_token", token);
        setIsLogged(true);
        await handlePostAuth();
        setSignInModalVisible(false);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("An unexpected error occurred.");
        }
      } finally {
        setLoading(false);
      }
    } else {
      const newError = {
        name: 0,
        email: 0,
        password: 0,
        confirmPassword: 0,
      };
      result.error.errors.forEach((err) => {
        if (err.path.includes("name")) newError.name = 1;
        if (err.path.includes("email")) newError.email = 1;
        if (err.path.includes("password")) newError.password = 1;
        if (err.path.includes("confirmPassword")) newError.confirmPassword = 1;
      });
      setRegisterErrors(newError);
    }
  };
  // Google sign-in handler using Firebase.
  const handleGoogleSignIn = async (isRegistration = false) => {
    try {
      const user = await signInWithGoogle();
      const email = user.email;
      const name = user.displayName;
      const res = await axios.post("http://127.0.0.1:8000/api/user/oauth", {
        email: email,
        emp_id: regEmployeeId,
        name: name,
        isRegistration: isRegistration,
      });
      const token = res.data.token;
      localStorage.setItem("access_token", token);
      setIsLogged(true);
      await handlePostAuth();
      setSignInModalVisible(false);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred.");
      }
    }
  };

  const handleTwitterSignIn = async (isRegistration = false) => {
    try {
      const user = await signInWithTwitter();
      const email = user.email;
      const name = user.displayName;
      const res = await axios.post("http://127.0.0.1:8000/api/user/oauth", {
        email: email,
        emp_id: regEmployeeId,
        name: name,
        isRegistration: isRegistration,
      });
      const token = res.data.token;
      localStorage.setItem("access_token", token);
      setIsLogged(true);
      await handlePostAuth();
      setSignInModalVisible(false);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred.");
      }
    }
  };

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

  return (
    <Dialog open={signInModalVisible} onOpenChange={handleOpenChange}>
      {/* Trigger button with hover animation */}
      <DialogTrigger asChild>
        <AnimatedButton className="bg-black border border-white/50 text-white cursor-pointer text-lg font-semibold px-10 py-4 h-auto rounded-full shadow-md hover:shadow-lg hover:bg-[#0a0a0a] hover:border-white/60 group relative overflow-hidden md:scale-90 lg:scale-100 scale-75">
          Get Started
        </AnimatedButton>
      </DialogTrigger>

      <DialogContent className="fixed sm:max-w-[425px] dark bg-[#131313] text-white border-none dialog-content top-5 p-0 ">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 to-teal-500"></div>

        <DialogHeader className="p-6 pb-2">
          <DialogTitle className="text-center text-2xl font-bold">
            {mode === "login" ? "Welcome Back" : "Join MoodPulse"}
          </DialogTitle>
        </DialogHeader>

        <div className="p-6">
          <AnimatePresence mode="wait">
            {showLoginForm && (
              <motion.div
                key="login-form"
                initial="hidden"
                animate="visible"
                exit="exit"
                variants={formVariants}
              >
                <form className="grid gap-4" onSubmit={handleLoginSubmit}>
                  <div className="flex flex-col gap-4">
                    <button
                      type="button"
                      onClick={() => handleGoogleSignIn()}
                      className="cursor-pointer relative w-full bg-white text-black hover:bg-gray-100 p-3 rounded-md flex items-center justify-center gap-2 font-medium transition-all duration-300 social-button"
                    >
                      <FcGoogle className="text-xl" />
                      <span>Sign in with Google</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => handleTwitterSignIn()}
                      className="cursor-pointer relative w-full bg-[#1DA1F2] text-white hover:bg-[#1a8cd8] p-3 rounded-md flex items-center justify-center gap-2 font-medium transition-all duration-300 social-button"
                    >
                      <FaTwitter className="text-xl" />
                      <span>Sign in with Twitter</span>
                    </button>

                    <div className="divider">
                      <span>Or continue with</span>
                    </div>

                    <div className="grid gap-2">
                      <Label
                        htmlFor="loginEmail"
                        className="text-sm font-medium"
                      >
                        Email
                      </Label>
                      <Input
                        id="loginEmail"
                        name="email"
                        type="email"
                        placeholder="you@example.com"
                        onChange={handleLoginChange}
                        required={true}
                        className="auth-input p-3 rounded-md"
                      />
                      {Loginerrors.email ? (
                        <p className="text-red-500 text-xs mt-1">
                          Enter valid email.
                        </p>
                      ) : null}
                    </div>

                    <div className="grid gap-2">
                      <Label
                        htmlFor="loginPassword"
                        className="text-sm font-medium"
                      >
                        Password
                      </Label>
                      <Input
                        id="loginPassword"
                        name="password"
                        type="password"
                        placeholder="••••••••"
                        onChange={handleLoginChange}
                        required={true}
                        className="auth-input p-3 rounded-md"
                      />
                      {Loginerrors.password ? (
                        <p className="text-red-500 text-xs mt-1">
                          Password must contain at least 8 characters, including
                          letters, numbers, and special characters
                        </p>
                      ) : null}
                    </div>
                  </div>

                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-md text-sm"
                    >
                      {error}
                    </motion.div>
                  )}

                  <DialogFooter className="mt-4">
                    <AnimatedButton
                      variant="default"
                      type="submit"
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
                          Signing In...
                        </span>
                      ) : (
                        "Sign In"
                      )}
                    </AnimatedButton>
                  </DialogFooter>

                  <p className="text-center text-sm mt-4">
                    Don&apos;t have an account?{" "}
                    <button
                      type="button"
                      className="text-emerald-500 font-medium cursor-pointer hover:text-emerald-400 transition-colors"
                      onClick={() => {
                        setError("");
                        setMode("register");
                        setRegisterStep("checkId");
                      }}
                    >
                      Create Account
                    </button>
                  </p>
                </form>
              </motion.div>
            )}

            {showVerifyForm && (
              <motion.div
                key="verify-form"
                initial="hidden"
                animate="visible"
                exit="exit"
                variants={formVariants}
              >
                <form className="grid gap-4" onSubmit={handleVerifyEmployeeId}>
                  <div className="grid gap-2">
                    <Label
                      htmlFor="regEmployeeId"
                      className="text-sm font-medium"
                    >
                      Employee ID
                    </Label>
                    <Input
                      id="regEmployeeId"
                      name="regEmployeeId"
                      type="text"
                      placeholder="Enter Employee ID"
                      value={regEmployeeId}
                      onChange={(e) => setRegEmployeeId(e.target.value)}
                      required={true}
                      className="auth-input p-3 rounded-md"
                    />
                    <p className="text-xs text-gray-400 mt-1">
                      Please enter your employee ID to verify your eligibility.
                    </p>
                  </div>

                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-md text-sm"
                    >
                      {error}
                    </motion.div>
                  )}

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
                          Verifying...
                        </span>
                      ) : (
                        "Verify Employee ID"
                      )}
                    </AnimatedButton>
                  </DialogFooter>

                  <p className="text-center text-sm mt-4">
                    Already have an account?{" "}
                    <button
                      type="button"
                      className="text-emerald-500 font-medium cursor-pointer hover:text-emerald-400 transition-colors"
                      onClick={() => {
                        setError("");
                        setMode("login");
                      }}
                    >
                      Sign In
                    </button>
                  </p>
                </form>
              </motion.div>
            )}

            {showRegisterForm && (
              <motion.div
                key="register-form"
                initial="hidden"
                animate="visible"
                exit="exit"
                variants={formVariants}
              >
                <div className="grid gap-4">
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 p-3 rounded-md text-sm text-center"
                  >
                    Employee ID verified successfully!
                  </motion.div>

                  <div className="flex flex-col gap-4">
                    <button
                      type="button"
                      onClick={() => handleGoogleSignIn(true)}
                      className="cursor-pointer relative w-full bg-white text-black hover:bg-gray-100 p-3 rounded-md flex items-center justify-center gap-2 font-medium transition-all duration-300 social-button"
                    >
                      <FcGoogle className="text-xl" />
                      <span>Register with Google</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => handleTwitterSignIn(true)}
                      className="cursor-pointer relative w-full bg-[#1DA1F2] text-white hover:bg-[#1a8cd8] p-3 rounded-md flex items-center justify-center gap-2 font-medium transition-all duration-300 social-button"
                    >
                      <FaTwitter className="text-xl" />
                      <span>Register with Twitter</span>
                    </button>

                    <div className="divider">
                      <span>Or continue with</span>
                    </div>
                  </div>

                  <form className="grid gap-4" onSubmit={handleRegisterSubmit}>
                    <div className="grid gap-2">
                      <Label htmlFor="regName" className="text-sm font-medium">
                        Name
                      </Label>
                      <Input
                        id="regName"
                        name="name"
                        type="text"
                        placeholder="Your Name"
                        onChange={handleRegisterChange}
                        required={true}
                        className="auth-input p-3 rounded-md"
                      />
                      {Registererrors.name ? (
                        <p className="text-red-500 text-xs mt-1">
                          Enter valid Name.
                        </p>
                      ) : null}
                    </div>

                    <div className="grid gap-2">
                      <Label htmlFor="regEmail" className="text-sm font-medium">
                        Email
                      </Label>
                      <Input
                        id="regEmail"
                        name="email"
                        type="email"
                        placeholder="you@example.com"
                        onChange={handleRegisterChange}
                        required={true}
                        className="auth-input p-3 rounded-md"
                      />
                      {Registererrors.email ? (
                        <p className="text-red-500 text-xs mt-1">
                          Enter valid email.
                        </p>
                      ) : null}
                    </div>

                    <div className="grid gap-2">
                      <Label
                        htmlFor="regPassword"
                        className="text-sm font-medium"
                      >
                        Password
                      </Label>
                      <Input
                        id="regPassword"
                        name="password"
                        type="password"
                        placeholder="••••••••"
                        onChange={handleRegisterChange}
                        required={true}
                        className="auth-input p-3 rounded-md"
                      />
                      {Registererrors.password ? (
                        <p className="text-red-500 text-xs mt-1">
                          Password must contain at least 8 characters, including
                          letters, numbers, and special characters
                        </p>
                      ) : null}
                    </div>

                    <div className="grid gap-2">
                      <Label
                        htmlFor="regConfirmPassword"
                        className="text-sm font-medium"
                      >
                        Confirm Password
                      </Label>
                      <Input
                        id="regConfirmPassword"
                        name="confirmPassword"
                        type="password"
                        placeholder="••••••••"
                        onChange={handleRegisterChange}
                        required={true}
                        className="auth-input p-3 rounded-md"
                      />
                      {Registererrors.confirmPassword ? (
                        <p className="text-red-500 text-xs mt-1">
                          Passwords Don&apos;t match{" "}
                        </p>
                      ) : null}
                    </div>

                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-md text-sm"
                      >
                        {error}
                      </motion.div>
                    )}

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
                            Registering...
                          </span>
                        ) : (
                          "Complete Registration"
                        )}
                      </AnimatedButton>
                    </DialogFooter>
                  </form>

                  <p className="text-center text-sm mt-4">
                    Already have an account?{" "}
                    <button
                      type="button"
                      className="text-emerald-500 font-medium cursor-pointer hover:text-emerald-400 transition-colors"
                      onClick={() => {
                        setError("");
                        setMode("login");
                      }}
                    >
                      Sign In
                    </button>
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </DialogContent>
    </Dialog>
  );
}
