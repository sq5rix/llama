import unittest
import tempfile
import os
from pathlib import Path
from contextual_rag import ContextualRAG


class TestContextualRAG(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for data and persistence
        self.temp_data_dir = tempfile.mkdtemp()
        self.temp_persist_dir = tempfile.mkdtemp()

        # Create some test files
        self.create_test_files()

        # Initialize RAG system
        self.rag = ContextualRAG(
            data_dir=self.temp_data_dir, persist_dir=self.temp_persist_dir
        )

    def tearDown(self):
        # Cleanup temporary directories
        import shutil

        shutil.rmtree(self.temp_data_dir)
        shutil.rmtree(self.temp_persist_dir)

    def create_test_files(self):
        # Create test file 1
        with open(os.path.join(self.temp_data_dir, "test1.txt"), "w") as f:
            f.write("This is a test document about artificial intelligence. " * 50)

        # Create test file 2
        with open(os.path.join(self.temp_data_dir, "test2.txt"), "w") as f:
            f.write("This document contains information about machine learning. " * 50)

    def test_context_size(self):
        size = self.rag.get_context_size()
        self.assertEqual(size, ContextualRAG.MAX_CONTEXT_SIZE)

    def test_load_and_split_files(self):
        chunks = self.rag.load_and_split_files()
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(len(chunks[0]), 3)  # (chunk, filename, position)

    def test_create_chunk_prefix(self):
        chunk = "This is a test chunk about AI."
        filename = "test.txt"
        position = 0

        prefix = self.rag.create_chunk_prefix(chunk, filename, position)
        self.assertIsInstance(prefix, str)
        self.assertTrue(len(prefix) > 0)

    def test_process_and_embed(self):
        self.rag.process_and_embed()
        self.assertIsNotNone(self.rag.vectorstore)
        self.assertTrue(len(self.rag.chunk_metadata) > 0)

    def test_infer(self):
        self.rag.process_and_embed()
        results = self.rag.infer("What is artificial intelligence?")

        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])
        self.assertIn("similarity_score", results[0])
        self.assertIn("metadata", results[0])


if __name__ == "__main__":
    unittest.main()
