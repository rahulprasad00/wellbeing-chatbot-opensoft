// components/ui/LoadingIndicator.tsx
// "use client";

// import React from "react";

// interface LoadingIndicatorProps {
//   text?: string;
//   className?: string;
// }

// const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({
//   text = "Loading content...",
//   className = "",
// }) => {
//   return (
//     <div className={`flex items-center justify-center p-6 ${className}`}>
//       <div className="flex flex-col items-center gap-3">
//         <div className="relative">
//           <div className="h-10 w-10 rounded-full border-2 border-white/20"></div>
//           <div className="absolute top-0 left-0 h-10 w-10 rounded-full border-2 border-t-[#00805A] border-r-[#86BC25] border-b-transparent border-l-transparent animate-spin"></div>
//         </div>
//         {text && <p className="text-white/70 text-sm">{text}</p>}
//       </div>
//     </div>
//   );
// };

// export default LoadingIndicator;