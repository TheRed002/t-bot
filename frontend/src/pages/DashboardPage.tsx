/**
 * Dashboard page component
 * Main overview page with trading metrics and charts
 */

import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
} from '@mui/material';
import { colors } from '@/theme/colors';

const DashboardPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Trading Dashboard
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Welcome to your T-Bot trading dashboard. Here you can monitor your portfolio performance,
        active bots, and market data in real-time.
      </Typography>

      <Grid container spacing={3}>
        {/* Portfolio Summary */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Value
              </Typography>
              <Typography variant="h4" sx={{ color: colors.financial.profit }}>
                $12,450.00
              </Typography>
              <Typography variant="body2" color="text.secondary">
                +$245.00 (2.01%) today
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Active Bots */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Bots
              </Typography>
              <Typography variant="h4" sx={{ color: colors.status.online }}>
                3
              </Typography>
              <Typography variant="body2" color="text.secondary">
                2 running, 1 paused
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Daily P&L */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Daily P&L
              </Typography>
              <Typography variant="h4" sx={{ color: colors.financial.profit }}>
                +$89.32
              </Typography>
              <Typography variant="body2" color="text.secondary">
                15 trades executed
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Level */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Level
              </Typography>
              <Typography variant="h4" sx={{ color: colors.financial.warning }}>
                Medium
              </Typography>
              <Typography variant="body2" color="text.secondary">
                3.2% portfolio exposure
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Chart placeholder */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, minHeight: 400 }}>
            <Typography variant="h6" gutterBottom>
              Portfolio Performance
            </Typography>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: 300,
                backgroundColor: colors.background.tertiary,
                borderRadius: 1,
              }}
            >
              <Typography color="text.secondary">
                Chart component will be implemented here
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;