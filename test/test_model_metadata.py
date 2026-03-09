"""Tests for ortpy.ModelMetadata."""
import pytest

import ortpy as ort


class TestModelMetadataFields:
    def test_producer_name(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.producer_name == "ortpy-test"

    def test_domain(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.domain == "test.domain"

    def test_description(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.description == "Test model description"

    def test_graph_description(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.graph_description == "Test graph description"

    def test_graph_name(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.graph_name == "identity_graph"

    def test_version(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.version == 42


class TestModelMetadataCustomMap:
    def test_custom_metadata_map(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        cmap = meta.custom_metadata_map
        assert isinstance(cmap, dict)
        assert cmap["key1"] == "value1"
        assert cmap["key2"] == "value2"

    def test_lookup_existing_key(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.lookup_custom_metadata("key1") == "value1"

    def test_lookup_nonexistent_key(self, metadata_model_path):
        session = ort.Session(str(metadata_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.lookup_custom_metadata("nonexistent") is None


class TestModelMetadataMinimal:
    """Test metadata on a model without explicit metadata fields."""

    def test_empty_producer(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.producer_name == ""

    def test_empty_domain(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.domain == ""

    def test_empty_custom_metadata(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        meta = session.get_model_metadata()
        assert meta.custom_metadata_map == {}
