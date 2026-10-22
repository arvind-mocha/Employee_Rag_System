import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from fastapi.testclient import TestClient

from app.api.main import app


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


class DataUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.data_directory = self.project_root / "data"
        self.structured_data_directory = self.project_root / "structured_data"

        self.patches = [
            patch("app.api.config.PROJECT_ROOT", self.project_root),
            patch("app.api.config.DATA_DIRECTORY", self.data_directory),
            patch(
                "app.api.config.STRUCTURED_DATA_DIRECTORY",
                self.structured_data_directory,
            ),
        ]
        for current_patch in self.patches:
            current_patch.start()

        self.client = TestClient(app)

    def tearDown(self) -> None:
        for current_patch in reversed(self.patches):
            current_patch.stop()
        self.temporary_directory.cleanup()

    def test_routes_and_flattens_files(self) -> None:
        zip_content = make_zip(
            {
                "documents/handbook.pdf": b"pdf content",
                "employees/list.csv": b"name\nAlex",
                "notes/readme.md": b"# Read me",
                "settings.json": b"{}",
            }
        )

        response = self.client.post(
            "/api/v1/data/upload",
            files={"zip_file": ("source-data.zip", zip_content, "application/zip")},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["total_files"], 4)
        self.assertTrue((self.data_directory / "handbook.pdf").is_file())
        self.assertTrue((self.data_directory / "readme.md").is_file())
        self.assertTrue((self.structured_data_directory / "list.csv").is_file())
        self.assertTrue((self.structured_data_directory / "settings.json").is_file())

    def test_rejects_a_non_zip_upload(self) -> None:
        response = self.client.post(
            "/api/v1/data/upload",
            files={"zip_file": ("source-data.zip", b"not a zip", "application/zip")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.data_directory.exists())

    def test_rejects_an_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/api/v1/data/upload",
            files={
                "zip_file": (
                    "source-data.zip",
                    make_zip({"image.png": b"image"}),
                    "application/zip",
                )
            },
        )

        self.assertEqual(response.status_code, 415)
        self.assertFalse(self.data_directory.exists())

    def test_does_not_overwrite_an_existing_file(self) -> None:
        self.data_directory.mkdir()
        existing_file = self.data_directory / "handbook.pdf"
        existing_file.write_bytes(b"original")

        response = self.client.post(
            "/api/v1/data/upload",
            files={
                "zip_file": (
                    "source-data.zip",
                    make_zip({"handbook.pdf": b"replacement"}),
                    "application/zip",
                )
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(existing_file.read_bytes(), b"original")

    def test_rejects_duplicate_names_after_flattening(self) -> None:
        response = self.client.post(
            "/api/v1/data/upload",
            files={
                "zip_file": (
                    "source-data.zip",
                    make_zip(
                        {
                            "team-a/profile.txt": b"first",
                            "team-b/profile.txt": b"second",
                        }
                    ),
                    "application/zip",
                )
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.data_directory.exists())


if __name__ == "__main__":
    unittest.main()
