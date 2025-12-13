"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import {
  getAuth,
  onAuthStateChanged,
  signOut,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  TwitterAuthProvider,
} from "firebase/auth";
import { app } from "@/firebase/config";
import { useRouter } from "next/navigation";
import { server } from "@/utils";
import axios from "axios";
import { toast } from "sonner";

const auth = getAuth(app);

export interface Employee {
  employee_id: string;
  employee_name: string;
  employee_email: string;
  role: string;
  is_selected: boolean;
  sentimental_score: number;
  shap_values: string[];
  is_Flagged: boolean;
  conversation_completed: boolean;
}

export interface HRUser {
  hr_id: string;
  hr_email: string;
  role: string;
  hr_name?: string;
}

interface AuthContextType {
  user: any;
  employeeData: Employee | null;
  hrData: HRUser | null;
  isLogged: boolean;
  signInModalVisible: boolean;
  setIsLogged: (isLogged: boolean) => void;
  setEmployeeData: (employee: Employee) => void;
  setHRData: (hrData: HRUser) => void;
  signIn: (email: string, password: string) => Promise<any>;
  signUp: (email: string, password: string, name: string) => Promise<any>;
  signInWithGoogle: () => Promise<any>;
  logout: () => Promise<void>;
  signInHR: (email: string, password: string) => Promise<any>;
  signInWithTwitter: () => Promise<any>;
  setSignInModalVisible: (visible: boolean) => void;
  fetchEmployeeProfile: () => Promise<Employee>;
  fetchHRProfile: () => Promise<HRUser>;
  check_role: (role: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<any>(null);
  const [isLogged, setIsLogged] = useState(false);
  const [employeeData, setEmployeeData] = useState<Employee | null>(null);
  const [hrData, setHRData] = useState<HRUser | null>(null);
  const [loading, setLoading] = useState(false);
  const [signInModalVisible, setSignInModalVisible] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token =
          typeof window !== "undefined" && localStorage.getItem("access_token");
        if (token) {
          setIsLogged(true);
          const decoded = JSON.parse(atob(token.split(".")[1]));
          console.log("Fetching HR Profile Based on role :", decoded.role);
          if (decoded.role === "hr") {
            await fetchHRProfile();
          } else if (decoded.role === "employee") {
            console.log(
              "Fetching Employee Profile Based on role :",
              decoded.role
            );
            const profile = await fetchEmployeeProfile();
            console.log("Employee Profile:", profile);

            if (profile.is_selected && profile.conversation_completed) {
              router.push("/conversation");
            } else router.push("/dashboard");
          }
        }
      } catch (error) {
        console.error("Error fetching user profile:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const check_role = (role: string) => {
    const token =
      typeof window !== "undefined" && localStorage.getItem("access_token");
    if (!token) return false;
    try {
      const decoded = JSON.parse(atob(token.split(".")[1]));
      if (decoded.role === role) {
        return true;
      }
      return false;
    } catch (error) {
      console.error("Invalid token:", error);
      return false;
    }
  };

  const signIn = async (email: string, password: string) => {
    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    setUser(userCredential.user);
    return userCredential;
  };

  const signUp = async (email: string, password: string, name: string) => {
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      email,
      password
    );
    setUser(userCredential.user);
    return userCredential;
  };

  const signInWithGoogle = async () => {
    const provider = new GoogleAuthProvider();
    const res = await signInWithPopup(auth, provider);
    setUser(res.user);
    return res.user;
  };

  const signInWithTwitter = async () => {
    const provider = new TwitterAuthProvider();
    const res = await signInWithPopup(auth, provider);
    setUser(res.user);
    return res.user;
  };

  const signInHR = async (email: string, password: string) => {
    try {
      const response = await fetch(`${server}/api/user/hr-login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error("Login failed");
      }

      const data = await response.json();
      localStorage.setItem("access_token", data.token);
      setIsLogged(true);
      // await fetchHRProfile();
      return data;
    } catch (error) {
      console.error("HR login failed:", error);
      throw error;
    }
  };

  const fetchEmployeeProfile = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const res = await axios.get(`${server}/api/user/employee`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      // console.log(res.data);
      const data = res.data;
      // setIsLogged(true);
      setEmployeeData(data);
      return data;
    } catch (error) {
      console.error("Error fetching employee profile:", error);
      throw error;
    }
  };

  const fetchHRProfile = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${server}/api/user/hr`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch HR profile");
      }

      const data = await response.json();
      const normalized: HRUser = {
        hr_id: data.hr_id || data.id?.toString() || "",
        hr_email: data.hr_email || data.email || "",
        role: data.role,
        hr_name: data.hr_name || data.name || data.email?.split("@")?.[0],
      };
      setHRData(normalized);
      return normalized;
    } catch (error) {
      console.error("Error fetching HR profile:", error);
      throw error;
    }
  };

  const logout = async () => {
    toast.loading("Logging out...");
    try {
      await signOut(auth);
      setUser(null);
      setIsLogged(false);
      localStorage.removeItem("access_token");
      toast.dismiss();
      toast.success("Logged out successfully");
      router.push("/");
    } catch (error) {
      toast.dismiss();
      toast.error("Failed to log out");
      console.error("Logout error:", error);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        employeeData,
        hrData,
        isLogged,
        signInModalVisible,
        setIsLogged,
        setEmployeeData,
        signInWithTwitter,
        signIn,
        signUp,
        signInWithGoogle,
        logout,
        setSignInModalVisible,
        signInHR,
        fetchEmployeeProfile,
        fetchHRProfile,
        setHRData,
        check_role,
      }}
    >
      {!loading ? children : <p>Loading...</p>}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
