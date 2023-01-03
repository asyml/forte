"""
Unit tests for multi pack related operations.
"""
import logging
from typing import Any, Dict
import unittest

from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack, MultiPackLink
from forte.data.ontology import Annotation, MultiPackGroup
from ft.onto.base_ontology import Token

logging.basicConfig(level=logging.DEBUG)


def _space_token(pack: DataPack):
    begin = 0
    for i, c in enumerate(pack.text):
        if c == " ":
            pack.add_entry(Token(pack, begin, i))
            begin = i + 1

    if begin < len(pack.text):
        pack.add_entry(Token(pack, begin, len(pack.text)))


class DataPackTest(unittest.TestCase):
    def setUp(self) -> None:
        # Note: input source is created automatically by the system, but we
        #  can also set it manually at test cases.
        self.multi_pack = MultiPack()
        self.data_pack1 = self.multi_pack.add_pack(ref_name="left pack")
        self.data_pack2 = self.multi_pack.add_pack(ref_name="right pack")

        self.data_pack1.pack_name = "some pack"
        self.data_pack1.set_text("This pack contains some sample data.")

        self.data_pack2.pack_name = "another pack"
        self.data_pack2.set_text("This pack contains some other sample data.")

    def test_serialization(self):
        mp: MultiPack = self.multi_pack

        # Serialize and deserialize MultiPack object
        serialized_mp = mp.to_string(drop_record=True)
        recovered_mp = MultiPack.from_string(serialized_mp)

        # Serialize and deserialize DataPack objects
        s_packs = [p.to_string() for p in mp.packs]
        recovered_packs = [DataPack.from_string(s) for s in s_packs]

        # Add the recovered DataPack objects to MultiPack
        recovered_mp.relink(recovered_packs)

        # Validate the deserialized MultiPack object
        for pack_name in mp.pack_names:
            self.assertEqual(
                recovered_mp.get_pack(pack_name).pack_id,
                mp.get_pack(pack_name).pack_id
            )

    def test_add_pack(self):
        data_pack3 = self.multi_pack.add_pack(ref_name="new pack")
        data_pack3.pack_name = "the third pack"
        data_pack3.set_text("Test to see if we can add new packs..")

        self.assertEqual(len(self.multi_pack.packs), 3)
        self.assertEqual(
            self.multi_pack.pack_names, ["left pack", "right pack", "new pack"]
        )

    def test_rename_pack(self):
        self.multi_pack.rename_pack("right pack", "last pack")
        self.assertEqual(self.multi_pack.pack_names, ["left pack", "last pack"])

    def test_multipack_groups(self):
        """
        Test some multi pack group.
        Returns:

        """
        # Add tokens to each pack.
        for pack in self.multi_pack.packs:
            _space_token(pack)

        # Create some group.
        token: Annotation
        left_tokens = {}
        for token in self.multi_pack.packs[0].get(Token):
            left_tokens[token.text] = token

        right_tokens = {}
        for token in self.multi_pack.packs[1].get(Token):
            right_tokens[token.text] = token

        for key, lt in left_tokens.items():
            if key in right_tokens:
                rt = right_tokens[key]
                self.multi_pack.add_entry(
                    MultiPackGroup(self.multi_pack, [lt, rt])
                )

        # Check the groups.
        expected_content = [
            ("This", "This"),
            ("pack", "pack"),
            ("contains", "contains"),
            ("some", "some"),
            ("sample", "sample"),
            ("data.", "data."),
        ]

        group_content = []
        g: MultiPackGroup
        for g in self.multi_pack.get(MultiPackGroup):
            e: Annotation
            temp_list = []
            for e in g.get_members():
                temp_list.append(e.text)
            group_content.append(tuple(temp_list))

        self.assertListEqual(expected_content, group_content)

        # Get raw groups
        group_content = []
        grp: Dict[str, Any]
        for grp in self.multi_pack.get(MultiPackGroup, get_raw=True):
            temp_list = []
            # Note here that grp represents a dictionary and not an object
            for pack, mem in grp['members']:
                mem_obj = self.multi_pack.get_subentry(pack, mem)
                temp_list.append(mem_obj.text)

            group_content.append(tuple(temp_list))
        self.assertListEqual(expected_content, group_content)
        


    def test_multipack_entries(self):
        """
        Test some multi pack entry.
        Returns:

        """
        # 1. Add tokens to each pack.
        for pack in self.multi_pack.packs:
            _space_token(pack)

        left_tokens = [t.text for t in self.multi_pack.packs[0].get(Token)]
        right_tokens = [t.text for t in self.multi_pack.packs[1].get(Token)]

        self.assertListEqual(
            left_tokens, ["This", "pack", "contains", "some", "sample", "data."]
        )
        self.assertListEqual(
            right_tokens,
            ["This", "pack", "contains", "some", "other", "sample", "data."],
        )

        # 2. Link the same words from two packs.
        token: Annotation
        left_tokens = {}
        for token in self.multi_pack.packs[0].get(Token):
            left_tokens[token.text] = token

        right_tokens = {}
        for token in self.multi_pack.packs[1].get(Token):
            right_tokens[token.text] = token

        for key, lt in left_tokens.items():
            if key in right_tokens:
                rt = right_tokens[key]
                self.multi_pack.add_entry(
                    MultiPackLink(self.multi_pack, lt, rt)
                )

        # One way to link tokens.
        linked_tokens = []
        for link in self.multi_pack.all_links:
            parent_text = link.get_parent().text
            child_text = link.get_child().text
            linked_tokens.append((parent_text, child_text))

        self.assertListEqual(
            linked_tokens,
            [
                ("This", "This"),
                ("pack", "pack"),
                ("contains", "contains"),
                ("some", "some"),
                ("sample", "sample"),
                ("data.", "data."),
            ],
        )

        # Another way to get the links
        linked_tokens = []
        for link in self.multi_pack.get(MultiPackLink):
            parent_text = link.get_parent().text
            child_text = link.get_child().text
            linked_tokens.append((parent_text, child_text))

        self.assertListEqual(
            linked_tokens,
            [
                ("This", "This"),
                ("pack", "pack"),
                ("contains", "contains"),
                ("some", "some"),
                ("sample", "sample"),
                ("data.", "data."),
            ],
        )

        # fix bug 559: additional test for index to pack_id changes
        serialized_mp = self.multi_pack.to_string(drop_record=False)
        recovered_mp = MultiPack.from_string(serialized_mp)
        s_packs = [p.to_string() for p in self.multi_pack.packs]
        recovered_packs = [DataPack.from_string(s) for s in s_packs]

        # 1st verify recovered_packs
        left_tokens_recovered = [t.text for t in recovered_packs[0].get(Token)]
        right_tokens_recovered = [t.text for t in recovered_packs[1].get(Token)]

        self.assertListEqual(
            left_tokens_recovered, ["This", "pack", "contains", "some", "sample", "data."]
        )
        self.assertListEqual(
            right_tokens_recovered,
            ["This", "pack", "contains", "some", "other", "sample", "data."],
        )

        recovered_mp.relink(recovered_packs)

        # teh verfiy the links are ok (restored correctly)
        linked_tokens_recovered = []
        for link in recovered_mp.all_links:
            parent_text = link.get_parent().text
            child_text = link.get_child().text
            linked_tokens_recovered.append((parent_text, child_text))

        self.assertListEqual(
            linked_tokens_recovered,
            [
                ("This", "This"),
                ("pack", "pack"),
                ("contains", "contains"),
                ("some", "some"),
                ("sample", "sample"),
                ("data.", "data."),
            ],
        )

        # 3. Test deletion

        # Delete the second link.
        self.multi_pack.delete_entry(self.multi_pack.links[1])

        linked_tokens = []
        for link in self.multi_pack.all_links:
            parent_text = link.get_parent().text
            child_text = link.get_child().text
            linked_tokens.append((parent_text, child_text))

        self.assertListEqual(
            linked_tokens,
            [
                ("This", "This"),
                ("contains", "contains"),
                ("some", "some"),
                ("sample", "sample"),
                ("data.", "data."),
            ],
        )

    def test_remove_pack(self):

        """
        1. Test to remove the added pack from multi_pack which is independent;
        2. Test to remove the added pack from multi_pack with MultiPackGroup and MultiPackLink;

        Returns:

        """

        data_pack10 = self.multi_pack.add_pack(ref_name="remove pack 10")
        data_pack11 = self.multi_pack.add_pack(ref_name="remove pack 11")
        data_pack12 = self.multi_pack.add_pack(ref_name="remove pack 12")
        data_pack13 = self.multi_pack.add_pack(ref_name="remove pack 13")

        ref_id10 = self.multi_pack.get_pack_index(data_pack10.pack_id)
        ref_id11 = self.multi_pack.get_pack_index(data_pack11.pack_id)
        ref_id12 = self.multi_pack.get_pack_index(data_pack12.pack_id)
        ref_id13 = self.multi_pack.get_pack_index(data_pack13.pack_id)

        ref_id1 = self.multi_pack.get_pack_index(self.data_pack1.pack_id)
        ref_id2 = self.multi_pack.get_pack_index(self.data_pack2.pack_id)

        data_pack10.pack_name = "the 1st pack for removing"
        data_pack10.set_text(
            "Test to see if we can remove the added pack which is independent"
        )

        data_pack11.pack_name = "the 2nd pack for removing"
        data_pack11.set_text(
            "Test to see if we can remove the added pack from MultiPackGroup and MultiPackLink"
        )

        data_pack12.pack_name = "the 3rd pack for removing"
        data_pack12.set_text(
            "Test to see if we can remove the added pack from MultiPackGroup and MultiPackLink"
        )

        data_pack13.pack_name = "the 4th pack for removing"
        data_pack13.set_text(
            "Test to see if we can remove the added pack from MultiPackGroup and MultiPackLink"
        )

        # Add MultiPackLink to data_pack11 and data_pack12 & Add MultiPackGroup to data_pack11, data_pack12
        # and data_pack13
        # Add tokens to each pack.
        for pack in self.multi_pack.packs[ref_id11: ref_id12 + 1]:
            _space_token(pack)

        # Create some group.
        token: Annotation

        remove_tokens_11 = {}
        for token in self.multi_pack.packs[ref_id11].get(Token):
            remove_tokens_11[token.text] = token

        remove_tokens_12 = {}
        for token in self.multi_pack.packs[ref_id12].get(Token):
            remove_tokens_12[token.text] = token

        remove_tokens_13 = {}
        for token in self.multi_pack.packs[ref_id13].get(Token):
            remove_tokens_13[token.text] = token

        for key, rt11 in remove_tokens_11.items():
            if key in remove_tokens_12:
                rt12 = remove_tokens_12[key]
                tmp_mpk_entry = MultiPackLink(self.multi_pack, rt11, rt12)
                self.multi_pack.add_entry(tmp_mpk_entry)  # Add Multi Pack Link
                # my_rt12 = remove_tokens_12[key]
                # tmp_mpk_entry2 = MultiPackLink(self.multi_pack, my_rt12, rt11)
                # self.multi_pack.add_entry(
                #     MultiPackLink(self.multi_pack, tmp_mpk_entry2, rt12) # Add Multi Pack Link to Link
                # )

                if key in remove_tokens_13:
                    rt13 = remove_tokens_13[key]
                    self.multi_pack.add_entry(
                        MultiPackGroup(
                            self.multi_pack, [rt11, rt12, rt13]
                        )  # Add Multi Pack Group of 3 packs
                    )

        check_list_id = [
            ref_id1,
            ref_id2,
            ref_id10,
            ref_id11,
            ref_id12,
            ref_id13,
        ]
        self.check_list_name = self.multi_pack.pack_names
        # print('check_list_id_all:', self.check_list_id)
        # print('check_list_name_all:', self.check_list_name)

        ## Preparation for remaining pack ID list check ##
        expected_id_list_1 = list(
            set(check_list_id) - set([ref_id10])
        )
        expected_id_list_2 = list(
            set(check_list_id)
            - set([ref_id10])
            - set([ref_id11])
        )
        expected_id_list_3 = list(
            set(check_list_id)
            - set([ref_id10])
            - set([ref_id11])
            - set([ref_id12])
        )
        ## Preparation for remaining pack name list check ##
        expected_name_list_1 = [
            "left pack",
            "right pack",
            "",
            "remove pack 11",
            "remove pack 12",
            "remove pack 13",
        ]
        expected_name_list_2 = [
            "left pack",
            "right pack",
            "",
            "",
            "remove pack 12",
            "remove pack 13",
        ]
        expected_name_list_3 = [
            "left pack",
            "right pack",
            "",
            "",
            "",
            "remove pack 13",
        ]
        expected_name_list_purge = [
            "left pack",
            "right pack",
            "remove pack 13",
        ]

        # Test to remove the data_pack10 which is independent
        self.multi_pack.remove_pack(ref_id10)
        ## remaining ref_id alignment check
        remaining_id_1 = [
            ref_id1,
            ref_id2,
            ref_id11,
            ref_id12,
            ref_id13,
        ]
        self.assertListEqual(expected_id_list_1, remaining_id_1)
        ## remaining pack name alignment check
        self.assertNotIn(["remove pack 10"], self.multi_pack.pack_names)
        self.assertListEqual(
            self.multi_pack.pack_names, expected_name_list_1
        )

        # Test to remove the added pack from multi_pack with MultiPackGroup
        self.multi_pack.remove_pack(ref_id11, True)
        ## remaining ref_id alignment check
        remaining_id_2 = [
            ref_id1,
            ref_id2,
            ref_id12,
            ref_id13,
        ]
        self.assertListEqual(expected_id_list_2, remaining_id_2)
        ## remaining pack name alignment check
        self.assertNotIn(["remove pack 11"], self.multi_pack.pack_names)
        self.assertListEqual(
            self.multi_pack.pack_names, expected_name_list_2
        )

        # Test to remove the added pack from multi_pack with MultiPackGroup and MultiPackLink
        self.multi_pack.remove_pack(ref_id12, True)
        ## remaining ref_id alignment check
        remaining_id_3 = [ref_id1, ref_id2, ref_id13]
        self.assertListEqual(expected_id_list_3, remaining_id_3)
        ## remaining pack name alignment check
        self.assertNotIn(["remove pack 12"], self.multi_pack.pack_names)
        self.assertListEqual(
            self.multi_pack.pack_names, expected_name_list_3
        )

        self.multi_pack.purge_deleted_packs()
        self.assertListEqual(
            self.multi_pack.pack_names, expected_name_list_purge
        )

    def test_remove_pack_auto_purge(self):

        """
        1. Test to remove the added pack from multi_pack which is independent;
        2. Test to remove the added pack from multi_pack with MultiPackGroup and MultiPackLink;

        Returns:

        """

        data_pack10 = self.multi_pack.add_pack(ref_name="remove pack 10")
        data_pack11 = self.multi_pack.add_pack(ref_name="remove pack 11")
        data_pack12 = self.multi_pack.add_pack(ref_name="remove pack 12")
        data_pack13 = self.multi_pack.add_pack(ref_name="remove pack 13")

        ref_id10 = self.multi_pack.get_pack_index(data_pack10.pack_id)
        ref_id11 = self.multi_pack.get_pack_index(data_pack11.pack_id)
        ref_id12 = self.multi_pack.get_pack_index(data_pack12.pack_id)
        ref_id13 = self.multi_pack.get_pack_index(data_pack13.pack_id)

        ref_id1 = self.multi_pack.get_pack_index(self.data_pack1.pack_id)
        ref_id2 = self.multi_pack.get_pack_index(self.data_pack2.pack_id)

        data_pack10.pack_name = "the 1st pack for removing"
        data_pack10.set_text(
            "Test to see if we can remove the added pack which is independent"
        )

        data_pack11.pack_name = "the 2nd pack for removing"
        data_pack11.set_text(
            "Test to see if we can remove the added pack from MultiPackGroup and MultiPackLink"
        )

        data_pack12.pack_name = "the 3rd pack for removing"
        data_pack12.set_text(
            "Test to see if we can remove the added pack from MultiPackGroup and MultiPackLink"
        )

        data_pack13.pack_name = "the 4th pack for removing"
        data_pack13.set_text(
            "Test to see if we can remove the added pack from MultiPackGroup and MultiPackLink"
        )

        # Add MultiPackLink to data_pack11 and data_pack12 & Add MultiPackGroup to data_pack11, data_pack12
        # and data_pack13
        # Add tokens to each pack.
        for pack in self.multi_pack.packs[ref_id11: ref_id12 + 1]:
            _space_token(pack)

        # Create some group.
        token: Annotation

        remove_tokens_11 = {}
        for token in self.multi_pack.packs[ref_id11].get(Token):
            remove_tokens_11[token.text] = token

        remove_tokens_12 = {}
        for token in self.multi_pack.packs[ref_id12].get(Token):
            remove_tokens_12[token.text] = token

        remove_tokens_13 = {}
        for token in self.multi_pack.packs[ref_id13].get(Token):
            remove_tokens_13[token.text] = token

        for key, rt11 in remove_tokens_11.items():
            if key in remove_tokens_12:
                rt12 = remove_tokens_12[key]
                tmp_mpk_entry = MultiPackLink(self.multi_pack, rt11, rt12)
                self.multi_pack.add_entry(tmp_mpk_entry)  # Add Multi Pack Link
                # my_rt12 = remove_tokens_12[key]
                # tmp_mpk_entry2 = MultiPackLink(self.multi_pack, my_rt12, rt11)
                # self.multi_pack.add_entry(
                #     MultiPackLink(self.multi_pack, tmp_mpk_entry2, rt12) # Add Multi Pack Link to Link
                # )

                if key in remove_tokens_13:
                    rt13 = remove_tokens_13[key]
                    self.multi_pack.add_entry(
                        MultiPackGroup(
                            self.multi_pack, [rt11, rt12, rt13]
                        )  # Add Multi Pack Group of 3 packs
                    )

        check_list_id = [
            ref_id1,
            ref_id2,
            ref_id10,
            ref_id11,
            ref_id12,
            ref_id13,
        ]
        self.check_list_name = self.multi_pack.pack_names
        # print('check_list_id_all:', self.check_list_id)
        # print('check_list_name_all:', self.check_list_name)

        ## Preparation for remaining pack ID list check ##
        expected_id_list_1 = list(
            set(check_list_id) - set([ref_id10])
        )
        expected_id_list_2 = list(
            set(check_list_id)
            - set([ref_id10])
            - set([ref_id11])
        )
        expected_id_list_3 = list(
            set(check_list_id)
            - set([ref_id10])
            - set([ref_id11])
            - set([ref_id12])
        )
        ## Preparation for remaining pack name list check ##
        expected_name_list_1 = [
            "left pack",
            "right pack",
            "",
            "remove pack 11",
            "remove pack 12",
            "remove pack 13",
        ]

        expected_name_list_purge = [
            "left pack",
            "right pack",
            "remove pack 13",
        ]

        # Test to remove the data_pack10 which is independent
        self.multi_pack.remove_pack(ref_id10)
        ## remaining ref_id alignment check
        remaining_id_1 = [
            ref_id1,
            ref_id2,
            ref_id11,
            ref_id12,
            ref_id13,
        ]
        self.assertListEqual(expected_id_list_1, remaining_id_1)
        ## remaining pack name alignment check
        self.assertNotIn(["remove pack 10"], self.multi_pack.pack_names)
        self.assertListEqual(
            self.multi_pack.pack_names, expected_name_list_1
        )

        # Test to remove the added pack from multi_pack with MultiPackGroup
        self.multi_pack.remove_pack(ref_id11, True)
        self.assertNotIn(["remove pack 11"], self.multi_pack.pack_names)

        # Test to remove the added pack (with MultiPackGroup and MultiPackLink)
        # and auto purge of lists
        self.multi_pack.remove_pack(ref_id12, True, True)
        self.assertNotIn(["remove pack 12"], self.multi_pack.pack_names)
        self.assertListEqual(
            self.multi_pack.pack_names, expected_name_list_purge
        )


if __name__ == "__main__":
    unittest.main()
