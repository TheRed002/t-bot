/**
 * Global notification system component
 * Displays toast notifications for the entire application
 */

import React, { useEffect } from 'react';
import {
  Snackbar,
  Alert,
  AlertTitle,
  IconButton,
  Box,
  Slide,
  SlideProps,
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '@/store';
import { removeNotification, clearNotifications } from '@/store/slices/uiSlice';
import { Notification } from '@/types';

// Slide transition for notifications
const SlideTransition = (props: SlideProps) => {
  return <Slide {...props} direction="left" />;
};

interface NotificationItemProps {
  notification: Notification;
  onClose: (id: string) => void;
}

const NotificationItem: React.FC<NotificationItemProps> = ({
  notification,
  onClose,
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
  }, [id, autoHide, duration, onClose]);

  const handleClose = () => {
    onClose(id);
  };

  return (
    <Snackbar
      open={true}
      anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      TransitionComponent={SlideTransition}
      sx={{ 
        position: 'relative',
        marginBottom: 1,
      }}
    >
      <Alert
        severity={type}
        variant="filled"
        action={
          <IconButton
            size="small"
            aria-label="close"
            color="inherit"
            onClick={handleClose}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        }
        sx={{
          minWidth: 350,
          maxWidth: 500,
          '& .MuiAlert-message': {
            width: '100%',
          },
        }}
      >
        {title && <AlertTitle>{title}</AlertTitle>}
        {message}
      </Alert>
    </Snackbar>
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
    <Box
      sx={{
        position: 'fixed',
        top: 20,
        right: 20,
        zIndex: (theme) => theme.zIndex.snackbar,
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        pointerEvents: 'none',
        '& > *': {
          pointerEvents: 'auto',
        },
      }}
    >
      {notifications.map((notification) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onClose={handleRemoveNotification}
        />
      ))}
      
      {/* Clear all button when there are multiple notifications */}
      {notifications.length > 2 && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'flex-end',
            marginTop: 1,
          }}
        >
          <IconButton
            size="small"
            onClick={handleClearAll}
            sx={{
              backgroundColor: 'action.hover',
              color: 'text.secondary',
              '&:hover': {
                backgroundColor: 'action.selected',
              },
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      )}
    </Box>
  );
};

export default NotificationSystem;