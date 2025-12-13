"use client";

import { useTheme } from "next-themes";
import { Toaster as Sonner } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme();

  return (
    <Sonner
      theme="dark" // Force dark theme for consistency
      className="toaster group custom-sonner"
      toastOptions={{
        classNames: {
          toast:
            "custom-toast group-[.error]:border-red-500 group-[.error]:bg-[#2a1a1a]",
          title: "custom-title group-[.error]:text-red-400",
          description: "custom-description group-[.error]:text-red-200",
          actionButton:
            "custom-action-button group-[.error]:bg-red-500 group-[.error]:hover:bg-red-600",
          cancelButton:
            "custom-cancel-button group-[.error]:bg-[#333] group-[.error]:text-red-200",
          closeButton: "custom-close-button group-[.error]:text-red-400",
        },
        duration: 4000,
      }}
      expand={false}
      closeButton={true}
      richColors={true}
      visibleToasts={3} // Limit number of visible toasts
      gap={8} // Add more gap between toasts
      offset={16} // Offset from the edge of the screen
      {...props}
    />
  );
};

export { Toaster };
