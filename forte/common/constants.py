# DataStore constants
# The index storing begin location in the internal entry data of DataStore.
BEGIN_INDEX = 0

# The index storing end location in the internal entry data of DataStore.
END_INDEX = 1

# The index storing tid in the internal entry data of DataStore.
TID_INDEX = 2

# The index storing entry type in the internal entry data of DataStore.
ENTRY_TYPE_INDEX = 3

# The index storing entry type (specific to Link and Group type). It is saved
# in the `tid_idx_dict` in DataStore.
ENTRY_DICT_TYPE_INDEX = 0

# The index storing entry index (specific to Link and Group type). It is saved
# in the `tid_idx_dict` in DataStore.
ENTRY_DICT_ENTRY_INDEX = 1

# The index storing parent entry tid in Link entries
PARENT_TID_INDEX = 0

# The index storing child entry tid in Link entries
CHILD_TID_INDEX = 1

# The index storing member entry type in Group entries
MEMBER_TYPE_INDEX = 0

# The index storing the list of member entries tid in Group entries
MEMBER_TID_INDEX = 1

# The index where the first attribute appears in the internal entry data of DataStore.
ATTR_BEGIN_INDEX = 4

# Name of the key to access the attribute dict of an entry type from
# ``_type_attributes`` of ``DataStore``.
ATTR_INFO_KEY = "attributes"

# Name of the key to access the type of an attribute from
# ``_type_attributes`` of ``DataStore``.
ATTR_TYPE_KEY = "type"

# Name of the key to access the index of an attribute from
# ``_type_attributes`` of ``DataStore``.
ATTR_INDEX_KEY = "index"

# Name of the key to access a set of parent names of an entry type from
# ``_type_attributes`` of ``DataStore``.
PARENT_CLASS_KEY = "parent_class"
