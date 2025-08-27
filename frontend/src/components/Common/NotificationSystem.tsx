/**
 * Global notification system component
 * Displays toast notifications for the entire application
 */

import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Shadcn/ui components
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';

// Lucide React icons
import {
  X,
  CheckCircle,
  AlertTriangle,
  Info,
  AlertCircle,
} from 'lucide-react';

import { useAppSelector, useAppDispatch } from '@/store';
import { removeNotification, clearNotifications } from '@/store/slices/uiSlice';
import { Notification } from '@/types';
import { cn } from '@/lib/utils';

interface NotificationItemProps {
  notification: Notification;
  onClose: (id: string) => void;
  index: number;
}

const NotificationItem: React.FC<NotificationItemProps> = ({
  notification,
  onClose,
  index,
}) => {
  const { id, type, title, message, autoHide, duration = 5000 } = notification;

  // Auto-hide notification after duration
  useEffect(() => {
    if (autoHide) {
      const timer = setTimeout(() => {
        onClose(id);
      }, duration);

      return () => clearTimeout(timer);
    }
    return undefined;
  }, [id, autoHide, duration, onClose]);

  const handleClose = () => {
    onClose(id);
  };

  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle className="h-4 w-4" />;
      case 'error':
        return <AlertCircle className="h-4 w-4" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4" />;
      case 'info':
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  const getVariant = () => {
    switch (type) {
      case 'error':
        return 'destructive';
      case 'warning':
        return 'default';
      case 'success':
        return 'default';
      case 'info':
      default:
        return 'default';
    }
  };

  const getColorClasses = () => {
    switch (type) {
      case 'success':
        return 'border-green-200 bg-green-50 text-green-800 dark:border-green-800 dark:bg-green-950 dark:text-green-400';
      case 'error':
        return 'border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-400';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50 text-yellow-800 dark:border-yellow-800 dark:bg-yellow-950 dark:text-yellow-400';
      case 'info':
      default:
        return 'border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-400';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 300, scale: 0.3 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 300, scale: 0.5, transition: { duration: 0.2 } }}
      transition={{ duration: 0.3, type: 'spring', damping: 25, stiffness: 500 }}
      style={{ zIndex: 9999 - index }}
      className="pointer-events-auto"
    >
      <Alert
        variant={getVariant()}
        className={cn(
          "relative w-80 pr-8 shadow-lg",
          getColorClasses()
        )}
      >
        {getIcon()}
        <div className="flex-1">
          {title && <AlertTitle className="text-sm font-semibold">{title}</AlertTitle>}
          {message && (
            <AlertDescription className="text-sm">
              {message}
            </AlertDescription>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="absolute right-2 top-2 h-6 w-6 p-0 hover:bg-transparent opacity-70 hover:opacity-100"
          onClick={handleClose}
        >
          <X className="h-3 w-3" />
          <span className="sr-only">Close</span>
        </Button>
      </Alert>
    </motion.div>
  );
};

const NotificationSystem: React.FC = () => {
  const dispatch = useAppDispatch();
  const notifications = useAppSelector((state) => state.ui.notifications);

  const handleRemoveNotification = (id: string) => {
    dispatch(removeNotification(id));
  };

  const handleClearAll = () => {
    dispatch(clearNotifications());
  };

  // Clear all notifications on page unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      dispatch(clearNotifications());
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [dispatch]);

  // Don't render if no notifications
  if (notifications.length === 0) {
    return null;
  }

  return (
    <div
      className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
      style={{ zIndex: 9999 }}
    >
      <AnimatePresence mode="popLayout">
        {notifications.map((notification, index) => (
          <NotificationItem
            key={notification.id}
            notification={notification}
            onClose={handleRemoveNotification}
            index={index}
          />
        ))}
      </AnimatePresence>
      
      {/* Clear all button when there are multiple notifications */}
      {notifications.length > 2 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="flex justify-end mt-2 pointer-events-auto"
        >
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearAll}
            className="bg-background/95 backdrop-blur-sm text-xs"
          >
            Clear All
          </Button>
        </motion.div>
      )}
    </div>
  );
};

export default NotificationSystem;