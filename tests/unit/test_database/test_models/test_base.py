"""
Unit tests for database base models and mixins.

This module tests all base model functionality including:
- TimestampMixin
- AuditMixin  
- MetadataMixin
- SoftDeleteMixin
"""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models.base import (
    AuditMixin,
    Base,
    MetadataMixin,
    SoftDeleteMixin,
    TimestampMixin,
)


class TestTimestampMixin:
    """Test TimestampMixin functionality."""

    def test_timestamp_mixin_attributes(self):
        """Test that TimestampMixin adds expected attributes."""
        
        class TestModel(TimestampMixin, Base):
            __tablename__ = "test_model"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Check that timestamp columns are added
        assert hasattr(TestModel, 'created_at')
        assert hasattr(TestModel, 'updated_at')
        
        # Check column properties
        created_at_col = TestModel.__table__.columns['created_at']
        updated_at_col = TestModel.__table__.columns['updated_at']
        
        assert created_at_col.nullable is False
        assert updated_at_col.nullable is False
        assert created_at_col.server_default is not None
        assert updated_at_col.server_default is not None
        assert updated_at_col.onupdate is not None

    def test_timestamp_mixin_instance_creation(self):
        """Test TimestampMixin instance creation."""
        
        class TestModel(TimestampMixin, Base):
            __tablename__ = "test_model_instance"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        
        # Timestamp fields exist but are None until saved to DB
        assert hasattr(instance, 'created_at')
        assert hasattr(instance, 'updated_at')
        assert instance.created_at is None  # Set by database
        assert instance.updated_at is None  # Set by database


class TestAuditMixin:
    """Test AuditMixin functionality."""

    def test_audit_mixin_attributes(self):
        """Test that AuditMixin adds expected attributes."""
        
        class TestModel(AuditMixin, Base):
            __tablename__ = "test_audit_model"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Check that audit columns are added (includes TimestampMixin)
        assert hasattr(TestModel, 'created_at')
        assert hasattr(TestModel, 'updated_at')
        assert hasattr(TestModel, 'created_by')
        assert hasattr(TestModel, 'updated_by')
        assert hasattr(TestModel, 'version')
        
        # Check column properties
        created_by_col = TestModel.__table__.columns['created_by']
        updated_by_col = TestModel.__table__.columns['updated_by']
        version_col = TestModel.__table__.columns['version']
        
        assert created_by_col.nullable is True  # Optional
        assert updated_by_col.nullable is True  # Optional
        assert version_col.nullable is False
        assert version_col.default.arg == 1

    def test_audit_mixin_instance_creation(self):
        """Test AuditMixin instance creation."""
        
        class TestModel(AuditMixin, Base):
            __tablename__ = "test_audit_instance"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test", created_by="user1", updated_by="user2")
        
        assert instance.created_by == "user1"
        assert instance.updated_by == "user2"
        # Version default is handled at database level
        assert hasattr(instance, 'version')

    def test_audit_mixin_version_default(self):
        """Test version field default value."""
        
        class TestModel(AuditMixin, Base):
            __tablename__ = "test_audit_version"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        
        # Version should have default value of 1
        version_col = TestModel.__table__.columns['version']
        assert version_col.default.arg == 1


class TestMetadataMixin:
    """Test MetadataMixin functionality."""

    def test_metadata_mixin_attributes(self):
        """Test that MetadataMixin adds expected attributes."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_metadata_model"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Check that metadata column is added
        assert hasattr(TestModel, 'metadata_json')
        
        # Check column properties
        metadata_col = TestModel.__table__.columns['metadata_json']
        assert metadata_col.default.arg == {}

    def test_metadata_mixin_instance_creation(self):
        """Test MetadataMixin instance creation."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_metadata_instance"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        
        assert hasattr(instance, 'metadata_json')
        # Methods should be available
        assert hasattr(instance, 'get_metadata')
        assert hasattr(instance, 'set_metadata')
        assert hasattr(instance, 'update_metadata')

    def test_get_metadata_with_data(self):
        """Test get_metadata method with existing data."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_get_metadata"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.metadata_json = {"key1": "value1", "key2": 42}
        
        assert instance.get_metadata("key1") == "value1"
        assert instance.get_metadata("key2") == 42
        assert instance.get_metadata("nonexistent") is None
        assert instance.get_metadata("nonexistent", "default") == "default"

    def test_get_metadata_no_data(self):
        """Test get_metadata method with no data."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_get_metadata_empty"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.metadata_json = None
        
        assert instance.get_metadata("key1") is None
        assert instance.get_metadata("key1", "default") == "default"

    def test_set_metadata_new_key(self):
        """Test set_metadata method with new key."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_set_metadata_new"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.metadata_json = None
        
        instance.set_metadata("key1", "value1")
        
        assert instance.metadata_json == {"key1": "value1"}
        assert instance.get_metadata("key1") == "value1"

    def test_set_metadata_existing_data(self):
        """Test set_metadata method with existing data."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_set_metadata_existing"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.metadata_json = {"existing": "value"}
        
        instance.set_metadata("new_key", "new_value")
        instance.set_metadata("existing", "updated_value")
        
        assert instance.metadata_json["new_key"] == "new_value"
        assert instance.metadata_json["existing"] == "updated_value"

    def test_update_metadata_new_dict(self):
        """Test update_metadata method with new metadata dict."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_update_metadata_new"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.metadata_json = None
        
        update_data = {"key1": "value1", "key2": 42}
        instance.update_metadata(update_data)
        
        assert instance.metadata_json == update_data
        assert instance.get_metadata("key1") == "value1"
        assert instance.get_metadata("key2") == 42

    def test_update_metadata_existing_dict(self):
        """Test update_metadata method with existing metadata dict."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_update_metadata_existing"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.metadata_json = {"existing": "value", "keep": "this"}
        
        update_data = {"existing": "updated", "new": "value"}
        instance.update_metadata(update_data)
        
        assert instance.metadata_json["existing"] == "updated"
        assert instance.metadata_json["keep"] == "this"
        assert instance.metadata_json["new"] == "value"


class TestSoftDeleteMixin:
    """Test SoftDeleteMixin functionality."""

    def test_soft_delete_mixin_attributes(self):
        """Test that SoftDeleteMixin adds expected attributes."""
        
        class TestModel(SoftDeleteMixin, Base):
            __tablename__ = "test_soft_delete_model"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Check that soft delete columns are added
        assert hasattr(TestModel, 'deleted_at')
        assert hasattr(TestModel, 'deleted_by')
        
        # Check column properties
        deleted_at_col = TestModel.__table__.columns['deleted_at']
        deleted_by_col = TestModel.__table__.columns['deleted_by']
        
        assert deleted_at_col.nullable is True  # Optional, None means not deleted
        assert deleted_by_col.nullable is True  # Optional

    def test_soft_delete_mixin_instance_creation(self):
        """Test SoftDeleteMixin instance creation."""
        
        class TestModel(SoftDeleteMixin, Base):
            __tablename__ = "test_soft_delete_instance"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        
        assert instance.deleted_at is None
        assert instance.deleted_by is None
        # Methods should be available
        assert hasattr(instance, 'is_deleted')
        assert hasattr(instance, 'soft_delete')
        assert hasattr(instance, 'restore')

    def test_is_deleted_property_false(self):
        """Test is_deleted property returns False for non-deleted record."""
        
        class TestModel(SoftDeleteMixin, Base):
            __tablename__ = "test_is_deleted_false"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.deleted_at = None
        
        assert instance.is_deleted is False

    def test_is_deleted_property_true(self):
        """Test is_deleted property returns True for deleted record."""
        
        class TestModel(SoftDeleteMixin, Base):
            __tablename__ = "test_is_deleted_true"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        instance.deleted_at = datetime.now(timezone.utc)
        
        assert instance.is_deleted is True

    def test_soft_delete_method_without_user(self):
        """Test soft_delete method without specifying user."""
        
        class TestModel(SoftDeleteMixin, Base):
            __tablename__ = "test_soft_delete_no_user"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        
        before_delete = datetime.now(timezone.utc)
        instance.soft_delete()
        after_delete = datetime.now(timezone.utc)
        
        assert instance.is_deleted is True
        assert before_delete <= instance.deleted_at <= after_delete
        assert instance.deleted_by is None

    def test_soft_delete_method_with_user(self):
        """Test soft_delete method with specifying user."""
        
        class TestModel(SoftDeleteMixin, Base):
            __tablename__ = "test_soft_delete_with_user"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        
        instance.soft_delete("admin_user")
        
        assert instance.is_deleted is True
        assert instance.deleted_at is not None
        assert instance.deleted_by == "admin_user"

    def test_restore_method(self):
        """Test restore method restores soft deleted record."""
        
        class TestModel(SoftDeleteMixin, Base):
            __tablename__ = "test_restore"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        instance = TestModel(name="test")
        
        # First delete the record
        instance.soft_delete("admin_user")
        assert instance.is_deleted is True
        
        # Then restore it
        instance.restore()
        
        assert instance.is_deleted is False
        assert instance.deleted_at is None
        assert instance.deleted_by is None


class TestMixinCombinations:
    """Test combinations of mixins working together."""

    def test_all_mixins_combined(self):
        """Test model with all mixins combined."""
        
        class TestModel(TimestampMixin, AuditMixin, MetadataMixin, SoftDeleteMixin, Base):
            __tablename__ = "test_all_mixins"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Check all attributes are present
        assert hasattr(TestModel, 'created_at')
        assert hasattr(TestModel, 'updated_at')
        assert hasattr(TestModel, 'created_by')
        assert hasattr(TestModel, 'updated_by')
        assert hasattr(TestModel, 'version')
        assert hasattr(TestModel, 'metadata_json')
        assert hasattr(TestModel, 'deleted_at')
        assert hasattr(TestModel, 'deleted_by')
        
        # Test instance creation
        instance = TestModel(
            name="test",
            created_by="user1",
            updated_by="user2"
        )
        
        # Test all mixin functionality works together
        instance.set_metadata("key", "value")
        instance.soft_delete("admin")
        
        assert instance.get_metadata("key") == "value"
        assert instance.is_deleted is True
        assert instance.deleted_by == "admin"

    def test_audit_includes_timestamp(self):
        """Test that AuditMixin includes TimestampMixin functionality."""
        
        class TestModel(AuditMixin, Base):
            __tablename__ = "test_audit_includes_timestamp"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # AuditMixin should include TimestampMixin
        assert hasattr(TestModel, 'created_at')
        assert hasattr(TestModel, 'updated_at')
        assert hasattr(TestModel, 'created_by')
        assert hasattr(TestModel, 'updated_by')
        assert hasattr(TestModel, 'version')

    def test_mixin_order_independence(self):
        """Test that mixin order doesn't affect functionality."""
        
        class TestModel1(MetadataMixin, SoftDeleteMixin, TimestampMixin, Base):
            __tablename__ = "test_order1"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        class TestModel2(TimestampMixin, SoftDeleteMixin, MetadataMixin, Base):
            __tablename__ = "test_order2"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Both should have the same attributes
        model1_attrs = set(TestModel1.__table__.columns.keys())
        model2_attrs = set(TestModel2.__table__.columns.keys())
        
        assert model1_attrs == model2_attrs
        
        # Both should work the same way
        instance1 = TestModel1(name="test1")
        instance2 = TestModel2(name="test2")
        
        instance1.set_metadata("key", "value1")
        instance2.set_metadata("key", "value2")
        
        instance1.soft_delete()
        instance2.soft_delete()
        
        assert instance1.get_metadata("key") == "value1"
        assert instance2.get_metadata("key") == "value2"
        assert instance1.is_deleted is True
        assert instance2.is_deleted is True


class TestBaseDeclarativeBase:
    """Test Base declarative base."""

    def test_base_is_declarative_base(self):
        """Test that Base is a proper declarative base."""
        from sqlalchemy.orm import DeclarativeMeta
        
        assert isinstance(Base, DeclarativeMeta)
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')

    def test_base_inheritance(self):
        """Test that models can inherit from Base."""
        
        class TestModel(Base):
            __tablename__ = "test_base_inheritance"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        assert issubclass(TestModel, Base)
        assert hasattr(TestModel, '__table__')
        assert TestModel.__tablename__ == "test_base_inheritance"

    def test_base_table_creation(self):
        """Test that Base can be used to create database tables."""
        
        class TestModel(Base):
            __tablename__ = "test_base_table"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Create in-memory SQLite database for testing
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        
        # Verify table was created
        assert "test_base_table" in Base.metadata.tables
        
        # Test basic CRUD operations
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create
        instance = TestModel(name="test")
        session.add(instance)
        session.commit()
        
        # Read
        retrieved = session.query(TestModel).filter_by(name="test").first()
        assert retrieved is not None
        assert retrieved.name == "test"
        
        session.close()


class TestMixinDatabaseIntegration:
    """Test mixin functionality with actual database operations."""

    def test_timestamp_mixin_with_database(self):
        """Test TimestampMixin with actual database operations."""
        
        class TestModel(TimestampMixin, Base):
            __tablename__ = "test_timestamp_db"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Create in-memory SQLite database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create instance
        instance = TestModel(name="test")
        assert instance.created_at is None  # Not set until saved
        
        # Save to database
        session.add(instance)
        session.commit()
        session.refresh(instance)
        
        # Timestamps should now be set by database
        assert instance.created_at is not None
        assert instance.updated_at is not None
        
        session.close()

    def test_metadata_mixin_serialization(self):
        """Test MetadataMixin JSON serialization with database."""
        
        class TestModel(MetadataMixin, Base):
            __tablename__ = "test_metadata_db"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))
        
        # Create in-memory SQLite database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create instance with metadata
        instance = TestModel(name="test")
        instance.set_metadata("key1", "value1")
        instance.set_metadata("key2", {"nested": "data"})
        
        # Save to database
        session.add(instance)
        session.commit()
        
        # Retrieve from database
        retrieved = session.query(TestModel).filter_by(name="test").first()
        
        assert retrieved.get_metadata("key1") == "value1"
        assert retrieved.get_metadata("key2") == {"nested": "data"}
        
        session.close()