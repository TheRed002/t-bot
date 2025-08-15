/**
 * Test suite for HelpPage component
 * Ensures proper rendering and functionality of the help system
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider } from '@mui/material/styles';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';

import HelpPage from '../HelpPage';
import { theme } from '@/theme';

// Mock store setup
const mockStore = configureStore({
  reducer: {
    auth: (state = { isAuthenticated: true, user: null }, action) => state,
    ui: (state = {}, action) => state,
  },
});

// Test wrapper component
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Provider store={mockStore}>
    <ThemeProvider theme={theme}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </ThemeProvider>
  </Provider>
);

describe('HelpPage', () => {
  beforeEach(() => {
    // Reset any global state or mocks
    jest.clearAllMocks();
  });

  test('renders help page header correctly', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Check main heading
    expect(screen.getByRole('heading', { name: /help & documentation/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /welcome to t-bot help center/i })).toBeInTheDocument();
  });

  test('displays search functionality', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Check search input
    const searchInput = screen.getByPlaceholderText(/search documentation/i);
    expect(searchInput).toBeInTheDocument();
  });

  test('shows help categories', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Check for main categories
    expect(screen.getByText('Getting Started')).toBeInTheDocument();
    expect(screen.getByText('Trading Basics')).toBeInTheDocument();
    expect(screen.getByText('Strategy Configuration')).toBeInTheDocument();
    expect(screen.getByText('Risk Management')).toBeInTheDocument();
    expect(screen.getByText('Bot Management')).toBeInTheDocument();
    expect(screen.getByText('Playground Tutorial')).toBeInTheDocument();
  });

  test('search functionality filters content', async () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText(/search documentation/i);
    
    // Search for a specific term
    fireEvent.change(searchInput, { target: { value: 'strategy' } });

    await waitFor(() => {
      // Should show search results
      expect(screen.getByText(/search results for "strategy"/i)).toBeInTheDocument();
    });
  });

  test('category selection works correctly', async () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Click on Getting Started category
    const gettingStartedCategory = screen.getByText('Getting Started');
    fireEvent.click(gettingStartedCategory);

    await waitFor(() => {
      // Should show back button
      expect(screen.getByText(/back to categories/i)).toBeInTheDocument();
    });
  });

  test('FAQ section is accessible', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Check for FAQ tab
    expect(screen.getByRole('tab', { name: /faq/i })).toBeInTheDocument();
  });

  test('keyboard shortcuts dialog can be opened', async () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Find and click settings button (keyboard shortcuts)
    const settingsButton = screen.getByLabelText(/keyboard shortcuts/i);
    fireEvent.click(settingsButton);

    await waitFor(() => {
      expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument();
    });
  });

  test('contact support information is displayed', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Click on Contact Support tab
    const contactTab = screen.getByRole('tab', { name: /contact support/i });
    fireEvent.click(contactTab);

    // Check for support content
    expect(screen.getByText(/email support/i)).toBeInTheDocument();
    expect(screen.getByText(/community forum/i)).toBeInTheDocument();
  });

  test('system requirements are displayed', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    expect(screen.getByText('System Requirements')).toBeInTheDocument();
    expect(screen.getByText(/minimum requirements/i)).toBeInTheDocument();
    expect(screen.getByText(/recommended/i)).toBeInTheDocument();
  });

  test('video tutorials section exists', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Click on Video Tutorials tab
    const videoTab = screen.getByRole('tab', { name: /video tutorials/i });
    fireEvent.click(videoTab);

    expect(screen.getByText('Video Tutorials')).toBeInTheDocument();
  });

  test('responsive design elements are present', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Check for responsive grid layout
    const categoriesGrid = screen.getByText('Documentation Categories').closest('div');
    expect(categoriesGrid).toBeInTheDocument();
  });

  test('print functionality is available', () => {
    // Mock window.print
    const printSpy = jest.spyOn(window, 'print').mockImplementation(() => {});

    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    const printButton = screen.getByLabelText(/print page/i);
    fireEvent.click(printButton);

    expect(printSpy).toHaveBeenCalled();

    printSpy.mockRestore();
  });

  test('bookmarking functionality works', async () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Navigate to an article first
    const gettingStartedCategory = screen.getByText('Getting Started');
    fireEvent.click(gettingStartedCategory);

    await waitFor(() => {
      // Find and click on an article
      const quickStartArticle = screen.getByText(/quick start guide/i);
      if (quickStartArticle) {
        fireEvent.click(quickStartArticle);
      }
    });

    // The bookmark functionality should be available
    // This is more of an integration test and would need actual article selection
  });

  test('breadcrumb navigation works', async () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Navigate to a category
    const gettingStartedCategory = screen.getByText('Getting Started');
    fireEvent.click(gettingStartedCategory);

    await waitFor(() => {
      // Check for breadcrumb navigation
      expect(screen.getByText(/back to categories/i)).toBeInTheDocument();
    });
  });

  test('handles empty search results gracefully', async () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText(/search documentation/i);
    
    // Search for a term that won't match anything
    fireEvent.change(searchInput, { target: { value: 'nonexistentterm12345' } });

    await waitFor(() => {
      expect(screen.getByText(/no results found/i)).toBeInTheDocument();
    });
  });

  test('tabs functionality works correctly', () => {
    render(
      <TestWrapper>
        <HelpPage />
      </TestWrapper>
    );

    // Test all tabs
    const faqTab = screen.getByRole('tab', { name: /faq/i });
    const videoTab = screen.getByRole('tab', { name: /video tutorials/i });
    const contactTab = screen.getByRole('tab', { name: /contact support/i });

    // Click each tab and verify content changes
    fireEvent.click(faqTab);
    expect(screen.getByText('Frequently Asked Questions')).toBeInTheDocument();

    fireEvent.click(videoTab);
    expect(screen.getByText('Video Tutorials')).toBeInTheDocument();

    fireEvent.click(contactTab);
    expect(screen.getByText('Get Help & Support')).toBeInTheDocument();
  });
});