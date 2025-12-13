import * as React from "react"

import { cn } from "@/lib/utils"

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "placeholder:text-muted-foreground  flex field-sizing-content min-w-0 rounded-md bg-transparent px-3 py-2 text-base disabled:cursor-not-allowed disabled:opacity-50 md:text-sm resize-none flex-1 min-h-[60px] max-h-[200px] text-foreground transition-all duration-300 outline-none  focus-visible:ring-0 ",
        className
      )}
      {...props}
    />
  )
}

export { Textarea }
