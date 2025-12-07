import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import { useHistory } from '@docusaurus/router';

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

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '80vh',
      padding: '20px'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '400px',
        padding: '2rem',
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        backgroundColor: 'white',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
      }}>
        <h2 style={{ marginBottom: '1.5rem', textAlign: 'center' }}>Sign Up</h2>

        {error && (
          <div style={{
            color: '#dc2626',
            backgroundColor: '#fee2e2',
            padding: '0.5rem',
            borderRadius: '4px',
            marginBottom: '1rem'
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="email" style={{ display: 'block', marginBottom: '0.5rem' }}>
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #e2e8f0',
                borderRadius: '4px',
              }}
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="password" style={{ display: 'block', marginBottom: '0.5rem' }}>
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
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #e2e8f0',
                borderRadius: '4px',
              }}
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="ros_experience_level" style={{ display: 'block', marginBottom: '0.5rem' }}>
              ROS Experience Level
            </label>
            <select
              id="ros_experience_level"
              name="ros_experience_level"
              value={formData.ros_experience_level}
              onChange={handleChange}
              required
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #e2e8f0',
                borderRadius: '4px',
              }}
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="focus_area" style={{ display: 'block', marginBottom: '0.5rem' }}>
              Focus Area
            </label>
            <select
              id="focus_area"
              name="focus_area"
              value={formData.focus_area}
              onChange={handleChange}
              required
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #e2e8f0',
                borderRadius: '4px',
              }}
            >
              <option value="hardware">Hardware</option>
              <option value="software">Software</option>
              <option value="both">Both</option>
            </select>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '0.75rem',
              backgroundColor: loading ? '#94a3b8' : '#4299e1',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            {loading ? 'Creating Account...' : 'Sign Up'}
          </button>
        </form>

        <div style={{ marginTop: '1rem', textAlign: 'center' }}>
          <p>
            Already have an account?{' '}
            <a href="/login" style={{ color: '#4299e1', textDecoration: 'underline' }}>
              Login here
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;