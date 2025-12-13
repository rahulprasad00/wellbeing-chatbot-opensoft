"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import AuthModal from "@/components/AuthModal";
import { useAuth } from "@/context/AuthContext";
import HrAuthModal from "@/components/HrAuthModal";
import Loader from "@/components/ui/Loader"; // Update the path to the correct location of the Loader component
// import ContentLoader from "@/components/ui/ContentLoader";
const Index: React.FC = () => {
  const [scrollY, setScrollY] = useState(0);
  const { signInModalVisible, setSignInModalVisible } = useAuth();
  const [isLoading, setIsLoading] = useState(true); // State to control loader visibility

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    window.addEventListener("scroll", handleScroll);

    // Simulate loading time and then hide the loader
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      clearTimeout(timer);
    };
  }, []);

  // Calculate opacity and transform values based on scroll position
  // const gradientOpacity = Math.max(0, Math.min(1, 1 - scrollY / 600));
  // const transformValue = `translateY(${Math.min(scrollY * 0.2, 100)}px)`;

  // Show full-screen loader while loading
  if (isLoading) {
    return <Loader fullScreen />;
  }

  return (
    <div className="w-full min-h-screen bg-black relative overflow-hidden">
      {/* Background image with scroll behavior */}
      <div
        className="w-full h-[800px] absolute left-0 top-0 pointer-events-none z-10 transition-all duration-300 ease-out"
        style={{
          backgroundImage: "url('/assets/Gradient.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          // opacity: gradientOpacity,
          // transform: transformValue,
        }}
      />

      {/* Content container */}
      <div className="max-w-[1440px] relative mx-auto ">
        <nav className="flex justify-between items-center py-6 z-10 px-6">
          <div className="flex items-center gap-2">
            <div className="h-10 w-10 bg-white/10 backdrop-blur-lg rounded-full flex items-center justify-center">
              <div className="h-4 w-4 rounded-full bg-gradient-to-br from-[#00805A] to-[#86BC25]"></div>
            </div>
            <span className="text-white text-xl font-semibold">
              MoodPulse
            </span>
          </div>

        </nav>

        <main className="flex flex-col min-h-screen vertical-align-middle gap-16 max-md:flex-col ml-4 sm:ml-6 lg:p-3 p-2">
          <div className="flex justify-between items-center lg:gap-16 gap-4 pt-0 pb-10 max-lg:flex-col min-h-[80vh]">
            <div className="flex flex-col items-start gap-8 max-w-[497px] z-10 sm:gap-3">
              <h1 className=" items-start flex flex-row text-[88px] font-normal  tracking-[-1.28px] text-white max-lg:text-[54px] max-sm:text-3xl sm:tracking-[-3.28px] sm:flex-col">
                <div>Cognition</div>
                <div className="sm:ml-0 ml-2">Pulse</div>
              </h1>
              <p className=" items-start text-white/95 text-base leading-[30px] font-medium lg:text-xl max-sm:text-base">
                Elevate your HR team with MoodPulse - an intelligent chatbot
                designed to monitor employee wellbeing, track mental health, and
                foster a positive workplace culture.
              </p>
              <div className="flex gap-10">
                <AuthModal />
                <HrAuthModal />
              </div>
            </div>

            {/* dashboard preview */}
            <div className="flex-1 max-w-[650px] relative lg:z-1 z-11 max-sm:scale-60 md:scale-100 scale-70">
              <div className="flex w-full lg:h-[650px] h-[350px] flex-col justify-between items-center border bg-[rgba(0,0,0,0.8)] p-4 rounded-3xl border-[rgba(255,255,255,0.12)] overflow-hidden shadow-2xl">
                <div className="flex lg:w-full w-[650px] h-full items-start bg-[rgba(30,30,30,0.9)] rounded-2xl  overflow-hidden ">
                  <Image
                    src="/dashboard-preview.jpg"
                    alt="Analytics Dashboard"
                    layout="fill"
                    objectFit="cover"
                    className=" w-full h-full  relative object-[5%_0%] overflow-hidden rounded-2xl "
                  />
                </div>
              </div>
            </div>
          </div>
        </main>

        <footer className="px-5 py-16 border-t border-white/10">
          <div className="max-w-[1170px] mx-auto">
            <div className="mx flex flex-col md:flex-row justify-between gap-10">
              <div className="flex flex-col gap-6">
                <div className="flex items-center gap-2">
                  <div className="h-8 w-8 bg-white/10 backdrop-blur-lg rounded-full flex items-center justify-center">
                    <div className="h-4 w-4 rounded-full bg-gradient-to-br from-[#00805A] to-[#86BC25]"></div>
                  </div>
                  <span className="text-white text-xl font-semibold">
                    MoodPulse
                  </span>
                </div>
                <p className="text-white/70 max-w-[340px]">
                  Empowering HR teams to build healthier workplaces through
                  AI-driven employee wellness monitoring.
                </p>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-10"></div>
            </div>

            <div className="mt-16  pt-6 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-4">
              <p className="text-white/50 text-sm">
                © 2023 MoodPulse. All rights reserved.
              </p>
              <div className="flex gap-6">
                <a
                  href="#"
                  className="text-white/60 hover:text-white transition-colors"
                >
                  {/* Social media icon */}
                </a>
                {/* More social media icons */}
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Index;