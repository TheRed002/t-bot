import * as React from "react"
import { cn } from "@/lib/utils"

interface ResizablePanelGroupProps {
  children: React.ReactNode
  direction?: "horizontal" | "vertical"
  className?: string
}

interface ResizablePanelProps {
  children: React.ReactNode
  defaultSize?: number
  minSize?: number
  maxSize?: number
  collapsible?: boolean
  className?: string
}

interface ResizeHandleProps {
  direction?: "horizontal" | "vertical"
  className?: string
}

export const ResizablePanelGroup: React.FC<ResizablePanelGroupProps> = ({
  children,
  direction = "horizontal",
  className,
}) => {
  return (
    <div
      className={cn(
        "flex h-full w-full",
        direction === "horizontal" ? "flex-row" : "flex-col",
        className
      )}
    >
      {children}
    </div>
  )
}

export const ResizablePanel: React.FC<ResizablePanelProps> = ({
  children,
  defaultSize = 50,
  minSize = 20,
  maxSize = 80,
  collapsible = false,
  className,
}) => {
  const [size, setSize] = React.useState(defaultSize)
  const [isCollapsed, setIsCollapsed] = React.useState(false)

  return (
    <div
      className={cn("relative flex flex-col", className)}
      style={{
        flex: isCollapsed ? "0 0 auto" : `${size} 1 0%`,
        minWidth: isCollapsed ? "auto" : `${minSize}%`,
        maxWidth: `${maxSize}%`,
      }}
    >
      {children}
    </div>
  )
}

const ResizeHandle: React.FC<ResizeHandleProps> = ({
  direction = "horizontal",
  className,
}) => {
  const [isDragging, setIsDragging] = React.useState(false)

  return (
    <div
      className={cn(
        "group relative flex items-center justify-center transition-colors",
        direction === "horizontal" 
          ? "w-1 cursor-col-resize hover:w-2 hover:bg-border" 
          : "h-1 cursor-row-resize hover:h-2 hover:bg-border",
        isDragging && "bg-border",
        className
      )}
      onMouseDown={() => setIsDragging(true)}
      onMouseUp={() => setIsDragging(false)}
      onMouseLeave={() => setIsDragging(false)}
    >
      <div
        className={cn(
          "absolute bg-border transition-all",
          direction === "horizontal" 
            ? "h-10 w-[1px] group-hover:w-[2px]" 
            : "w-10 h-[1px] group-hover:h-[2px]"
        )}
      />
    </div>
  )
}

// Simple implementation without complex resizing logic
// For production, consider using react-resizable-panels library
export { ResizablePanelGroup as PanelGroup, ResizablePanel as Panel, ResizeHandle }