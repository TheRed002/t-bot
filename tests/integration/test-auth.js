#!/usr/bin/env node

/**
 * Test authentication flow
 */

const http = require('http');

function testLogin() {
  const data = JSON.stringify({
    username: 'admin',
    password: 'admin123'
  });

  const options = {
    hostname: 'localhost',
    port: 8000,
    path: '/api/v1/auth/login',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  };

  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let responseData = '';

      res.on('data', (chunk) => {
        responseData += chunk;
      });

      res.on('end', () => {
        try {
          const parsed = JSON.parse(responseData);
          console.log('Status Code:', res.statusCode);
          console.log('Response:', JSON.stringify(parsed, null, 2));
          
          if (res.statusCode === 200 && parsed.success) {
            console.log('\n✅ Authentication successful!');
            console.log('User:', parsed.user?.username);
            console.log('Token present:', !!parsed.token);
            console.log('Token type:', parsed.token?.token_type);
          } else {
            console.log('\n❌ Authentication failed');
          }
          
          resolve(parsed);
        } catch (e) {
          console.error('Failed to parse response:', e);
          console.log('Raw response:', responseData);
          reject(e);
        }
      });
    });

    req.on('error', (error) => {
      console.error('Request failed:', error.message);
      reject(error);
    });

    req.write(data);
    req.end();
  });
}

// Run the test
console.log('Testing authentication endpoint...\n');
testLogin().catch(console.error);