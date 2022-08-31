# DataStore constants
# The name of the attribute storing the begin location in the internal
# entry data of DataStore.
BEGIN_ATTR_NAME = "begin"

# The name of the attribute storing the end location in the internal
# entry data of DataStore.
END_ATTR_NAME = "end"

# The index storing tid in the internal entry data of DataStore.
TID_INDEX = 0

# The index storing entry type in the internal entry data of DataStore.
ENTRY_TYPE_INDEX = 1

# The name of the attribute storing the payload index location in the
# internal entry data of DataStore.
PAYLOAD_ID_ATTR_NAME = "payload_idx"

# The index storing entry type (specific to Link and Group type). It is saved
# in the `tid_idx_dict` in DataStore.
ENTRY_DICT_TYPE_INDEX = 0

# The index storing entry index (specific to Link and Group type). It is saved
# in the `tid_idx_dict` in DataStore.
ENTRY_DICT_ENTRY_INDEX = 1

# The name of the attribute storing the parent entry tid in Link entries
PARENT_TID_ATTR_NAME = "parent"

# The name of the attribute storing the parent entry type in Link entries
PARENT_TYPE_ATTR_NAME = "parent_type"

# The name of the attribute storing the child entry tid in Link entries
CHILD_TID_ATTR_NAME = "child"

# The name of the attribute storing the child entry type in Link entries
CHILD_TYPE_ATTR_NAME = "child_type"

# The name of the attribute storing the member entry type in Group entries
MEMBER_TYPE_ATTR_NAME = "member_type"

# The name of the attribute storing the list of member entries tid in Group entries
MEMBER_TID_ATTR_NAME = "members"

# The index where the first attribute appears in the internal entry data of DataStore.
ATTR_BEGIN_INDEX = 2

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

# Name of the class field in JSON serialization schema of BasePack
JSON_CLASS_FIELD = "_json_class"

# Name of the state field in JSON serialization schema of BasePack
JSON_STATE_FIELD = "_json_state"
