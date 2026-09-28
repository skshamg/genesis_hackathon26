import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-sm text-sm font-semibold transition disabled:pointer-events-none disabled:opacity-50",
  { variants: {
    variant: {
      default: "border border-primary bg-primary text-primary-foreground hover:bg-primary/85",
      outline: "border border-border bg-transparent text-foreground hover:border-primary hover:text-primary",
      ghost: "text-muted-foreground hover:bg-accent hover:text-foreground",
      link: "text-primary underline-offset-4 hover:underline",
      destructive: "border border-negative bg-negative text-foreground",
      secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
    },
    size: { default: "h-10 px-4", sm: "h-8 px-3 text-xs", lg: "h-11 px-6", icon: "size-9" },
  }, defaultVariants: { variant: "default", size: "default" } },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> { asChild?: boolean }
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({className,variant,size,type="button",asChild: _asChild,...props},ref)=><button ref={ref} type={type} className={cn(buttonVariants({variant,size}),className)} {...props}/>);
Button.displayName="Button";