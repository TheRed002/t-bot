# T-Bot Trading System Frontend

A professional React-based frontend for the T-Bot cryptocurrency trading system, built with TypeScript and Material-UI.

## 🚀 Features

- **Modern React 18** with TypeScript for type safety
- **Material-UI (MUI)** with custom dark theme optimized for financial trading
- **Redux Toolkit** for predictable state management
- **React Query** for efficient API data fetching and caching
- **WebSocket integration** for real-time trading updates
- **Responsive design** that works on desktop and tablet
- **Performance optimized** with lazy loading and code splitting
- **Comprehensive testing** setup with Jest and Testing Library

## 🏗️ Architecture

### Core Technologies
- **React 18** - Latest React with concurrent features
- **TypeScript** - Strong typing for better developer experience
- **Material-UI v5** - Modern React component library
- **Redux Toolkit** - Simplified Redux for state management
- **React Query** - Powerful data synchronization for React
- **React Router v6** - Declarative routing
- **Socket.IO Client** - Real-time bidirectional communication

### Project Structure
```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Common/         # Common components (Loading, Notifications)
│   │   ├── Dashboard/      # Dashboard-specific components
│   │   ├── BotManagement/  # Bot management components
│   │   ├── Portfolio/      # Portfolio components
│   │   ├── Charts/         # Chart components
│   │   └── Layout/         # Layout components
│   ├── pages/              # Page components
│   ├── services/           # API services and utilities
│   │   └── api/           # API client and endpoints
│   ├── store/              # Redux store configuration
│   │   ├── slices/        # Redux slices
│   │   └── middleware/    # Custom middleware
│   ├── hooks/              # Custom React hooks
│   ├── utils/              # Utility functions
│   ├── theme/              # MUI theme configuration
│   └── types/              # TypeScript type definitions
├── public/                 # Static assets
└── tests/                  # Test files
```

## 🎨 Design System

### Color Palette
- **Primary**: Professional blue (`#1c96ff`) for branding
- **Financial Colors**: Green for profits, red for losses
- **Status Colors**: Semantic colors for different states
- **Dark Theme**: Optimized for reduced eye strain during trading

### Typography
- **Primary Font**: Inter (modern and readable)
- **Monospace Font**: JetBrains Mono (for numerical data)
- **Responsive scaling** across different screen sizes

### Components
- **Cards**: Elevated surfaces for data display
- **Charts**: Integrated chart components for financial data
- **Forms**: Consistent form styling across the application
- **Navigation**: Sidebar and header navigation

## 🔧 Development

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation
```bash
cd frontend
npm install
```

### Development Server
```bash
npm start
```
Runs the app in development mode on http://localhost:3000

### Build for Production
```bash
npm run build
```
Builds the app for production to the `dist` folder

### Testing
```bash
npm test          # Run tests
npm run test:watch # Run tests in watch mode
npm run test:coverage # Run tests with coverage
```

### Linting and Formatting
```bash
npm run lint      # Run ESLint
npm run lint:fix  # Fix ESLint issues
npm run type-check # Run TypeScript type checking
```

## 📱 Pages and Features

### Dashboard
- Portfolio overview with real-time metrics
- Active bot status monitoring
- Performance charts and analytics
- Quick action buttons

### Bot Management
- Create, configure, and manage trading bots
- Real-time bot status and performance tracking
- Start/stop/pause controls
- Bot configuration forms

### Portfolio
- Real-time position tracking
- Balance overview across exchanges
- P&L analysis with charts
- Historical performance data

### Strategy Center
- Browse available trading strategies
- Strategy configuration and customization
- Backtesting interface
- Performance comparison tools

### Risk Dashboard
- Real-time risk metrics monitoring
- Circuit breaker status and controls
- Alert management system
- Risk limit configuration

## 🔐 Authentication

- JWT-based authentication
- Automatic token refresh
- Protected routes
- Session management

## ⚡ Real-time Features

- WebSocket connection for live data
- Real-time price updates
- Bot status changes
- Portfolio updates
- Risk alerts and notifications

## 🧪 Testing Strategy

### Unit Tests
- Component testing with React Testing Library
- Redux slice testing
- Utility function testing
- Custom hook testing

### Integration Tests
- API integration testing
- User flow testing
- Navigation testing

### Performance Testing
- Bundle size monitoring
- Render performance testing
- Memory leak detection

## 🚀 Performance Optimizations

- **Code Splitting**: Lazy loading of routes and components
- **Memoization**: React.memo and useMemo for expensive operations
- **Virtualization**: For large data tables and lists
- **Image Optimization**: Optimized asset loading
- **Bundle Analysis**: Webpack bundle optimization

## 🔧 Configuration

### Environment Variables
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=http://localhost:8000
NODE_ENV=development|production
```

### API Configuration
- Axios client with interceptors
- Automatic retry logic
- Error handling
- Request/response logging

## 📦 Deployment

### Docker Support
```dockerfile
# Production build
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Nginx Configuration
- Static file serving
- API proxy configuration
- GZIP compression
- Security headers

## 🤝 Contributing

1. Follow the established code style
2. Write tests for new features
3. Update documentation as needed
4. Use conventional commit messages

## 📈 Monitoring

- Performance monitoring with React DevTools
- Error tracking with Error Boundaries
- Analytics for user interactions
- Real-time system health monitoring

---

Built with ❤️ for professional cryptocurrency trading