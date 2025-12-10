import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import { useHistory } from '@docusaurus/router';
import Layout from '@theme/Layout';

const SignupPage = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    ros_experience_level: 'beginner',
    focus_area: 'both',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { signup } = useAuth();
  const { showToast } = useToast();
  const history = useHistory();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Map form data to the expected backend format
    const userData = {
      email: formData.email,
      password: formData.password,
      ros_experience_level: formData.ros_experience_level,
      focus_area: formData.focus_area,
      specialization: [], // Default empty array
      language_preference: 'en', // Default language
    };

    const result = await signup(userData);

    if (result.success) {
      showToast('✓ Account created successfully!', 'success');
      setTimeout(() => {
        history.push('/'); // Redirect to home after signup
      }, 500);
    } else {
      setError(result.error);
      showToast('✗ Signup failed', 'error');
    }

    setLoading(false);
  };

  const inputStyle = {
    width: '100%',
    padding: '0.875rem 1rem',
    border: '1px solid rgba(66, 153, 225, 0.2)',
    borderRadius: '12px',
    background: 'rgba(15, 23, 42, 0.8)',
    color: '#F1F5F9',
    fontSize: '14px',
    fontFamily: 'Inter, sans-serif',
    transition: 'all 0.2s',
    outline: 'none',
  };

  const labelStyle = {
    display: 'block',
    marginBottom: '0.5rem',
    color: '#F1F5F9',
    fontSize: '14px',
    fontWeight: 600,
  };

  return (
    <Layout title="Sign Up" description="Create an account to access the textbook">
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '80vh',
        padding: '40px 20px',
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Animated background grid */}
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `
            linear-gradient(rgba(66, 153, 225, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(66, 153, 225, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          animation: 'gridMove 20s linear infinite',
          transform: 'rotateX(60deg) translateZ(-100px)',
          transformStyle: 'preserve-3d',
          opacity: 0.3,
        }} />

        <div style={{
          width: '100%',
          maxWidth: '500px',
          padding: '3rem',
          border: '1px solid rgba(66, 153, 225, 0.2)',
          borderRadius: '20px',
          background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%)',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(66, 153, 225, 0.1)',
          backdropFilter: 'blur(20px)',
          position: 'relative',
          zIndex: 1,
        }}>
          {/* Decorative robot icon */}
          <div style={{
            textAlign: 'center',
            marginBottom: '2rem',
          }}>
            <div style={{
              fontSize: '48px',
              marginBottom: '1rem',
            }}>🤖</div>
            <h2 style={{
              marginBottom: '0.5rem',
              fontSize: '32px',
              fontWeight: 700,
              background: 'linear-gradient(135deg, #FFFFFF 0%, #4299E1 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}>
              Join RoboLearn
            </h2>
            <p style={{
              color: '#94A3B8',
              fontSize: '14px',
            }}>
              Start your journey in robotics and AI
            </p>
          </div>

          {error && (
            <div style={{
              color: '#FCA5A5',
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              padding: '0.875rem',
              borderRadius: '12px',
              marginBottom: '1.5rem',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              fontSize: '14px',
            }}>
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '1.25rem' }}>
              <label htmlFor="email" style={labelStyle}>
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                style={inputStyle}
                onFocus={(e) => {
                  e.target.style.borderColor = '#4299E1';
                  e.target.style.boxShadow = '0 0 0 3px rgba(66, 153, 225, 0.1)';
                  e.target.style.background = 'rgba(15, 23, 42, 1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(66, 153, 225, 0.2)';
                  e.target.style.boxShadow = 'none';
                  e.target.style.background = 'rgba(15, 23, 42, 0.8)';
                }}
              />
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label htmlFor="password" style={labelStyle}>
                Password
              </label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                minLength="8"
                placeholder="Minimum 8 characters"
                style={inputStyle}
                onFocus={(e) => {
                  e.target.style.borderColor = '#4299E1';
                  e.target.style.boxShadow = '0 0 0 3px rgba(66, 153, 225, 0.1)';
                  e.target.style.background = 'rgba(15, 23, 42, 1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(66, 153, 225, 0.2)';
                  e.target.style.boxShadow = 'none';
                  e.target.style.background = 'rgba(15, 23, 42, 0.8)';
                }}
              />
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label htmlFor="ros_experience_level" style={labelStyle}>
                ROS Experience Level
              </label>
              <select
                id="ros_experience_level"
                name="ros_experience_level"
                value={formData.ros_experience_level}
                onChange={handleChange}
                required
                style={inputStyle}
                onFocus={(e) => {
                  e.target.style.borderColor = '#4299E1';
                  e.target.style.boxShadow = '0 0 0 3px rgba(66, 153, 225, 0.1)';
                  e.target.style.background = 'rgba(15, 23, 42, 1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(66, 153, 225, 0.2)';
                  e.target.style.boxShadow = 'none';
                  e.target.style.background = 'rgba(15, 23, 42, 0.8)';
                }}
              >
                <option value="beginner" style={{ background: '#0F172A' }}>Beginner</option>
                <option value="intermediate" style={{ background: '#0F172A' }}>Intermediate</option>
                <option value="advanced" style={{ background: '#0F172A' }}>Advanced</option>
              </select>
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <label htmlFor="focus_area" style={labelStyle}>
                Focus Area
              </label>
              <select
                id="focus_area"
                name="focus_area"
                value={formData.focus_area}
                onChange={handleChange}
                required
                style={inputStyle}
                onFocus={(e) => {
                  e.target.style.borderColor = '#4299E1';
                  e.target.style.boxShadow = '0 0 0 3px rgba(66, 153, 225, 0.1)';
                  e.target.style.background = 'rgba(15, 23, 42, 1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(66, 153, 225, 0.2)';
                  e.target.style.boxShadow = 'none';
                  e.target.style.background = 'rgba(15, 23, 42, 0.8)';
                }}
              >
                <option value="hardware" style={{ background: '#0F172A' }}>Hardware</option>
                <option value="software" style={{ background: '#0F172A' }}>Software</option>
                <option value="both" style={{ background: '#0F172A' }}>Both</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              style={{
                width: '100%',
                padding: '0.875rem',
                background: loading
                  ? 'rgba(148, 163, 184, 0.5)'
                  : 'linear-gradient(135deg, #4299E1 0%, #667EEA 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: '16px',
                fontWeight: 600,
                transition: 'all 0.3s',
                boxShadow: loading ? 'none' : '0 4px 20px rgba(66, 153, 225, 0.4)',
              }}
              onMouseOver={(e) => {
                if (!loading) {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 6px 24px rgba(66, 153, 225, 0.6)';
                }
              }}
              onMouseOut={(e) => {
                if (!loading) {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = '0 4px 20px rgba(66, 153, 225, 0.4)';
                }
              }}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                  <span style={{
                    width: '16px',
                    height: '16px',
                    border: '2px solid rgba(255, 255, 255, 0.3)',
                    borderTop: '2px solid white',
                    borderRadius: '50%',
                    animation: 'spin 0.8s linear infinite',
                  }} />
                  Creating Account...
                </span>
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          <div style={{
            marginTop: '2rem',
            textAlign: 'center',
            paddingTop: '1.5rem',
            borderTop: '1px solid rgba(66, 153, 225, 0.1)',
          }}>
            <p style={{ color: '#94A3B8', fontSize: '14px' }}>
              Already have an account?{' '}
              <a
                href="/login"
                style={{
                  color: '#4299E1',
                  textDecoration: 'none',
                  fontWeight: 600,
                  transition: 'color 0.2s',
                }}
                onMouseOver={(e) => e.target.style.color = '#667EEA'}
                onMouseOut={(e) => e.target.style.color = '#4299E1'}
              >
                Login here
              </a>
            </p>
          </div>
        </div>

        <style>{`
          @keyframes gridMove {
            0% { transform: rotateX(60deg) translateZ(-100px) translateY(0); }
            100% { transform: rotateX(60deg) translateZ(-100px) translateY(50px); }
          }
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          input::placeholder,
          select option {
            color: #64748B;
          }
        `}</style>
      </div>
    </Layout>
  );
};

export default SignupPage;