"""Tests for FastAPI endpoints"""
import pytest
from unittest.mock import Mock, patch


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_success(self, test_client, sample_query_request):
        """Test successful query returns expected response structure"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test answer about course content."
        assert len(data["sources"]) == 2

    def test_query_creates_session_when_not_provided(self, test_client):
        """Test that a new session is created when session_id is not provided"""
        response = test_client.post("/api/query", json={"query": "test query"})

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_query_uses_provided_session(self, test_client, sample_query_request_with_session):
        """Test that provided session_id is used"""
        response = test_client.post("/api/query", json=sample_query_request_with_session)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-789"

    def test_query_returns_sources_with_links(self, test_client, sample_query_request):
        """Test that sources include text and optional links"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        sources = data["sources"]
        assert len(sources) == 2
        assert sources[0]["text"] == "Building Towards Computer Use - Lesson 3"
        assert sources[0]["link"] == "https://example.com/lesson3"

    def test_query_empty_results(self, test_client_empty):
        """Test query with no matching results"""
        response = test_client_empty.post(
            "/api/query",
            json={"query": "something not in any course"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "couldn't find" in data["answer"].lower()
        assert data["sources"] == []

    def test_query_missing_query_field(self, test_client):
        """Test that missing query field returns validation error"""
        response = test_client.post("/api/query", json={})

        assert response.status_code == 422

    def test_query_empty_query_string(self, test_client):
        """Test query with empty string"""
        response = test_client.post("/api/query", json={"query": ""})

        # Empty string is technically valid, the RAG system handles it
        assert response.status_code == 200

    def test_query_internal_error(self, test_client_error):
        """Test that internal errors return 500 status"""
        response = test_client_error.post(
            "/api/query",
            json={"query": "test query"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_success(self, test_client):
        """Test successful course stats retrieval"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3

    def test_get_courses_returns_titles(self, test_client):
        """Test that course titles are returned correctly"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "Building Towards Computer Use" in data["course_titles"]
        assert "Introduction to MCP Servers" in data["course_titles"]

    def test_get_courses_empty(self, test_client_empty):
        """Test courses endpoint with no courses loaded"""
        response = test_client_empty.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_internal_error(self, test_client_error):
        """Test that internal errors return 500 status"""
        response = test_client_error.get("/api/courses")

        assert response.status_code == 500
        assert "Failed to fetch analytics" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET / endpoint"""

    def test_root_returns_api_info(self, test_client):
        """Test root endpoint returns API information"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestRequestValidation:
    """Tests for request validation"""

    def test_query_with_invalid_json(self, test_client):
        """Test that invalid JSON returns error"""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_with_wrong_content_type(self, test_client):
        """Test that wrong content type is handled"""
        response = test_client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422

    def test_query_with_extra_fields(self, test_client):
        """Test that extra fields are ignored"""
        response = test_client.post(
            "/api/query",
            json={
                "query": "test query",
                "extra_field": "should be ignored",
                "another_field": 123
            }
        )

        assert response.status_code == 200


class TestResponseFormat:
    """Tests for response format compliance"""

    def test_query_response_content_type(self, test_client, sample_query_request):
        """Test that response has correct content type"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.headers["content-type"] == "application/json"

    def test_courses_response_content_type(self, test_client):
        """Test that courses response has correct content type"""
        response = test_client.get("/api/courses")

        assert response.headers["content-type"] == "application/json"

    def test_query_response_source_structure(self, test_client, sample_query_request):
        """Test that each source has required fields"""
        response = test_client.post("/api/query", json=sample_query_request)

        data = response.json()
        for source in data["sources"]:
            assert "text" in source
            # link is optional but should be present in schema
            assert "link" in source or source.get("link") is None
