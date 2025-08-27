/**
 * Test suite for HelpPage component
 * Ensures proper rendering and functionality of the help system
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import HelpPage from '../HelpPage';

// Mock the theme
jest.mock('@/theme', () => ({
  theme: {
    colors: {
      primary: '#DC2626',
    },
  },
}));

describe('HelpPage', () => {
  const renderWithRouter = (component: React.ReactElement) => {
    return render(
      <BrowserRouter>
        {component}
      </BrowserRouter>
    );
  };

  it('renders without crashing', () => {
    renderWithRouter(<HelpPage />);
    expect(screen.getByText(/help/i)).toBeInTheDocument();
  });

  it('displays the main heading', () => {
    renderWithRouter(<HelpPage />);
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('displays documentation sections', () => {
    renderWithRouter(<HelpPage />);
    expect(screen.getByText(/getting started/i)).toBeInTheDocument();
  });
});