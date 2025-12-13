"use client";

import React from "react";

interface LoaderProps {
  fullScreen?: boolean;
}

const Loader: React.FC<LoaderProps> = ({ fullScreen = true }) => {
  if (fullScreen) {
    return (
      <div
        className="fixed inset-0 bg-black flex flex-col items-center justify-center z-50"
        style={{
          background:
            "linear-gradient(135deg, rgba(0,0,0,0.97) 0%, rgba(0,20,10,0.95) 100%)",
        }}
      >
        <div className="relative flex flex-col items-center">
          {/* Main loader circle */}
          <div className="relative h-24 w-24">
            {/* Outer spinning gradient ring */}
            <div
              className="absolute inset-0 rounded-full border-4 border-transparent animate-spin"
              style={{
                borderTopColor: "#00805A",
                borderRightColor: "#86BC25",
                animationDuration: "1.5s",
              }}
            ></div>

            {/* Inner pulsing circle */}
            <div className="absolute inset-3 rounded-full bg-black flex items-center justify-center">
              <div
                className="h-10 w-10 rounded-full animate-pulse"
                style={{
                  background:
                    "linear-gradient(135deg, #00805A 0%, #86BC25 100%)",
                  animationDuration: "1.5s",
                }}
              ></div>
            </div>
          </div>

          {/* Logo text */}
          <div className="mt-8 flex items-center gap-2">
            <div className="h-6 w-6 bg-white/10 backdrop-blur-lg rounded-full flex items-center justify-center">
              <div className="h-3 w-3 rounded-full bg-gradient-to-br from-[#00805A] to-[#86BC25]"></div>
            </div>
            <span className="text-white text-xl font-semibold">MoodPulse</span>
          </div>

          {/* Loading text */}
          <p className="mt-4 text-white/70 text-sm font-medium">
            Loading your Bot...
          </p>
        </div>
      </div>
    );
  }

  // For inline use (non-fullscreen)
  return (
    <div className="flex items-center justify-center p-4">
      <div className="relative h-12 w-12">
        <div
          className="absolute inset-0 rounded-full border-3 border-transparent animate-spin"
          style={{
            borderTopColor: "#00805A",
            borderRightColor: "#86BC25",
            animationDuration: "1.2s",
          }}
        ></div>
        <div className="absolute inset-2 rounded-full bg-black/80 flex items-center justify-center">
          <div
            className="h-5 w-5 rounded-full animate-pulse"
            style={{
              background: "linear-gradient(135deg, #00805A 0%, #86BC25 100%)",
            }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default Loader;
