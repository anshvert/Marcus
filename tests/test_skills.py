import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marcus.skills.store import SkillStore


class SkillTests(unittest.TestCase):
    def test_discovery_does_not_load_body_and_read_is_bounded_to_skill_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes").mkdir()
            (root / "notes" / "SKILL.md").write_text("---\nname: notes\ndescription: Work with personal notes.\n---\n\nPrivate workflow body")
            store = SkillStore(root)
            self.assertNotIn("Private", store.catalog())
            self.assertIn("Private", store.read("notes")["instructions"])
            self.assertEqual(len(store.discover("personal")), 1)
            with self.assertRaises(ValueError):
                store.read("../elsewhere")

    def test_symlink_cannot_load_external_instructions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            skills = root / "skills"
            skills.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (outside / "SKILL.md").write_text("description: external")
            (skills / "escape").symlink_to(outside)
            self.assertEqual(SkillStore(skills).discover(), [])
            with self.assertRaises(ValueError):
                SkillStore(skills).read("escape")
