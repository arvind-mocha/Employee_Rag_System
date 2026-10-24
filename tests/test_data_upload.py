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
        self.ingestion_runner_patch = patch(
            "app.api.controllers.data.ingestion_runner"
        )
        for current_patch in self.patches:
            current_patch.start()
        self.ingestion_runner = self.ingestion_runner_patch.start()

        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.ingestion_runner_patch.stop()
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
        self.assertEqual(response.json()["ignored_files"], [])
        self.assertEqual(response.json()["renamed_files"], {})
        self.assertTrue((self.data_directory / "handbook.pdf").is_file())
        self.assertTrue((self.data_directory / "readme.md").is_file())
        self.assertTrue((self.structured_data_directory / "list.csv").is_file())
        self.assertTrue((self.structured_data_directory / "settings.json").is_file())
        self.ingestion_runner.mark_data_changed.assert_called_once_with()

    def test_rejects_a_non_zip_upload(self) -> None:
        response = self.client.post(
            "/api/v1/data/upload",
            files={"zip_file": ("source-data.zip", b"not a zip", "application/zip")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.data_directory.exists())
        self.ingestion_runner.mark_data_changed.assert_not_called()

    def test_rejects_an_archive_without_supported_files(self) -> None:
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

        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.data_directory.exists())
        self.ingestion_runner.mark_data_changed.assert_not_called()

    def test_ignores_unsupported_files_in_a_mixed_archive(self) -> None:
        response = self.client.post(
            "/api/v1/data/upload",
            files={
                "zip_file": (
                    "source-data.zip",
                    make_zip(
                        {
                            "documents/handbook.pdf": b"pdf content",
                            "images/logo.png": b"image",
                        }
                    ),
                    "application/zip",
                )
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["total_files"], 1)
        self.assertEqual(response.json()["ignored_files"], ["images/logo.png"])
        self.assertTrue((self.data_directory / "handbook.pdf").is_file())
        self.assertFalse((self.data_directory / "logo.png").exists())
        self.ingestion_runner.mark_data_changed.assert_called_once_with()

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

        self.assertEqual(response.status_code, 201)
        self.assertEqual(existing_file.read_bytes(), b"original")
        self.assertEqual(
            (self.data_directory / "handbook_1.pdf").read_bytes(),
            b"replacement",
        )
        self.assertEqual(
            response.json()["renamed_files"],
            {"handbook.pdf": "handbook_1.pdf"},
        )

    def test_renames_duplicate_names_after_flattening(self) -> None:
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

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            (self.data_directory / "profile.txt").read_bytes(),
            b"first",
        )
        self.assertEqual(
            (self.data_directory / "profile_1.txt").read_bytes(),
            b"second",
        )
        self.assertEqual(
            response.json()["renamed_files"],
            {"team-b/profile.txt": "profile_1.txt"},
        )

    def test_uses_the_next_available_numeric_suffix(self) -> None:
        self.data_directory.mkdir()
        (self.data_directory / "handbook.pdf").write_bytes(b"original")
        (self.data_directory / "handbook_1.pdf").write_bytes(b"first duplicate")

        response = self.client.post(
            "/api/v1/data/upload",
            files={
                "zip_file": (
                    "source-data.zip",
                    make_zip({"handbook.pdf": b"second duplicate"}),
                    "application/zip",
                )
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            (self.data_directory / "handbook_2.pdf").read_bytes(),
            b"second duplicate",
        )

    def test_rejects_an_unsafe_path_even_when_its_type_is_unsupported(self) -> None:
        response = self.client.post(
            "/api/v1/data/upload",
            files={
                "zip_file": (
                    "source-data.zip",
                    make_zip({"../image.png": b"image", "handbook.pdf": b"pdf"}),
                    "application/zip",
                )
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.data_directory.exists())


if __name__ == "__main__":
    unittest.main()
