import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors duration-300 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive relative overflow-hidden",
  {
    variants: {
      variant: {
        default:
          "bg-gradient-to-r from-emerald-600 to-teal-700 text-white hover:from-emerald-500 hover:to-teal-600",
        destructive:
          "bg-gradient-to-r from-red-600 to-rose-700 text-white hover:from-red-500 hover:to-rose-600",
        outline:
          "border border-gray-200/30 bg-gray-900/60 backdrop-blur-sm text-white hover:bg-gray-800/70 hover:border-white/40",
        secondary:
          "bg-gradient-to-r from-gray-800 to-gray-900 text-white hover:from-gray-700 hover:to-gray-800",
        ghost:
          "text-white hover:bg-white/10 hover:backdrop-blur-sm hover:text-white",
        link: "text-emerald-400 underline-offset-4 hover:underline hover:text-emerald-300",
      },
      size: {
        default: "h-10 px-5 py-2 has-[>svg]:px-4",
        sm: "h-9 rounded-md gap-1.5 px-4 has-[>svg]:px-3",
        lg: "h-12 rounded-md px-8 has-[>svg]:px-6 text-base",
        icon: "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)


export function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"
  
  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

// Enhanced Animated Button with improved ripple effect
export function AnimatedButton({
  className,
  variant,
  size,
  asChild = false,
  children,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"
  const [ripples, setRipples] = React.useState<Array<{ id: number; x: number; y: number; size: number }>>([])
  const nextId = React.useRef(0)
  
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    const button = event.currentTarget
    const rect = button.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const size = Math.max(rect.width, rect.height) * 2.5
    
    // Add new ripple
    const id = nextId.current
    nextId.current += 1
    setRipples([...ripples, { id, x, y, size }])
    
    // Remove ripple after animation completes
    setTimeout(() => {
      setRipples(current => current.filter(ripple => ripple.id !== id))
    }, 800)
    
    if (props.onClick) {
      props.onClick(event)
    }
  }

  return (
    <Comp
      data-slot="button"
      className={cn(
        buttonVariants({ variant, size, className }),
        "transition-all duration-300 ease-out group"
      )}
      onClick={handleClick}
      {...props}
    >
      {ripples.map(ripple => (
        <span
          key={ripple.id}
          className="absolute bg-white/30 rounded-full pointer-events-none animate-ripple"
          style={{
            left: ripple.x - ripple.size / 2,
            top: ripple.y - ripple.size / 2,
            width: ripple.size,
            height: ripple.size,
          }}
        />
      ))}
      <span className="relative z-10 flex items-center justify-center gap-2 ">
        {children}
      </span>
    </Comp>
  )
}

// Add this to your global CSS
export const globalCss = `/* Animation classes moved to global.css */`;

// Enhanced MoodPulse branded button with improved gradient effects
export function GradientButton({
  className,
  size,
  asChild = false,
  children,
  ...props
}: React.ComponentProps<"button"> &
  Omit<VariantProps<typeof buttonVariants>, "variant"> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"
  const [ripples, setRipples] = React.useState<Array<{ id: number; x: number; y: number; size: number }>>([])
  const nextId = React.useRef(0)
  
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    const button = event.currentTarget
    const rect = button.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const size = Math.max(rect.width, rect.height) * 2.5
    
    // Add new ripple
    const id = nextId.current
    nextId.current += 1
    setRipples([...ripples, { id, x, y, size }])
    
    // Remove ripple after animation completes
    setTimeout(() => {
      setRipples(current => current.filter(ripple => ripple.id !== id))
    }, 800)
    
    if (props.onClick) {
      props.onClick(event)
    }
  }

  return (
    <Comp
      data-slot="button"
      className={cn(
        buttonVariants({ size }),
        "relative overflow-hidden bg-gradient-to-r from-teal-600 to-emerald-700 text-white transition-all duration-300 ease-out border-0 hover:shadow-xl hover:shadow-emerald-500/30 hover:scale-[1.02] active:scale-[0.98] hover:from-teal-500 hover:to-emerald-600 shadow-lg shimmer",
        className
      )}
      onClick={handleClick}
      {...props}
    >
      {ripples.map(ripple => (
        <span
          key={ripple.id}
          className="absolute bg-white/30 rounded-full pointer-events-none animate-ripple"
          style={{
            left: ripple.x - ripple.size / 2,
            top: ripple.y - ripple.size / 2,
            width: ripple.size,
            height: ripple.size,
          }}
        />
      ))}
      <span className="relative z-10 flex items-center justify-center gap-2 ">
        {children}
      </span>
    </Comp>
  )
}

// Maintaining the original MoodPulseButton in case it's being referenced
export function MoodPulseButton({
  className,
  size,
  variant = "primary",
  asChild = false,
  children,
  ...props
}: React.ComponentProps<"button"> &
  Omit<VariantProps<typeof buttonVariants>, "variant"> & {
    asChild?: boolean,
    variant?: "primary" | "secondary" | "ghost"
  }) {
  const Comp = asChild ? Slot : "button"
  const [ripples, setRipples] = React.useState<Array<{ id: number; x: number; y: number; size: number }>>([])
  const nextId = React.useRef(0)
  
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    const button = event.currentTarget
    const rect = button.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const size = Math.max(rect.width, rect.height) * 2.5
    
    // Add new ripple
    const id = nextId.current
    nextId.current += 1
    setRipples([...ripples, { id, x, y, size }])
    
    // Remove ripple after animation completes
    setTimeout(() => {
      setRipples(current => current.filter(ripple => ripple.id !== id))
    }, 800)
    
    if (props.onClick) {
      props.onClick(event)
    }
  }

  const getButtonClasses = () => {
    const baseClasses = "relative overflow-hidden transition-all duration-300 ease-out border-0 shadow-lg group focus-visible:ring-2 focus-visible:ring-emerald-500/50 focus-visible:ring-offset-1";
    
    if (variant === "primary") {
      return cn(
        baseClasses,
        "bg-gradient-to-r from-teal-600 to-emerald-700 text-white hover:shadow-xl hover:shadow-emerald-500/30 hover:scale-[1.02] active:scale-[0.98] hover:from-teal-500 hover:to-emerald-600",
      );
    } else if (variant === "secondary") {
      return cn(
        baseClasses,
        "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/15 hover:border-white/30 hover:shadow-xl hover:shadow-emerald-500/20 hover:scale-[1.02] active:scale-[0.98]",
      );
    } else {
      return cn(
        baseClasses,
        "bg-transparent text-white hover:bg-white/10 hover:shadow-sm hover:scale-[1.02] active:scale-[0.98]",
      );
    }
  };

  return (
    <Comp
      data-slot="button"
      className={cn(
        buttonVariants({ size }),
        getButtonClasses(),
        "shimmer",
        className
      )}
      onClick={handleClick}
      {...props}
    >
      {ripples.map(ripple => (
        <span
          key={ripple.id}
          className="absolute bg-white/30 rounded-full pointer-events-none animate-ripple"
          style={{
            left: ripple.x - ripple.size / 2,
            top: ripple.y - ripple.size / 2,
            width: ripple.size,
            height: ripple.size,
          }}
        />
      ))}
      <span className="relative z-10 flex items-center justify-center gap-2 ">
        {children}
      </span>
    </Comp>
  )
}

export { buttonVariants }