"""BrainGlobe atlas selection widget"""

from config import AVAILABLE_ATLASES
from PyQt5 import QtWidgets, QtCore
from services.atlas import Atlas



class AtlasWidget(QtWidgets.QWidget):
    """
    Widget for selecting and searching BrainGlobe atlases and structures.
    """

    structures_selected = QtCore.pyqtSignal(list)

    def __init__(self, parent = None):
        super().__init__(parent)
        self.setStyleSheet("background-color: lightblue;")
        self.atlas_service = Atlas()
        self._setup_ui()
        self._load_default_atlas()


    def _setup_ui(self) -> None:
        """
        Docstring for _setup_ui
        
        :param self: Description
        """

        layout = QtWidgets.QVBoxLayout(self)

        # Atlas Selection 

        atlas_row = QtWidgets.QHBoxLayout()
        atlas_row.addWidget(QtWidgets.QLabel("Select Atlas:"))

        self.atlas_combo = QtWidgets.QComboBox()
        self.atlas_combo.addItems(AVAILABLE_ATLASES)
        self.atlas_combo.currentTextChanged.connect(self._on_atlas_changed)
        atlas_row.addWidget(self.atlas_combo)

        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search structures...")
        self.search_box.textChanged.connect(self._on_search_text_changed)
        atlas_row.addWidget(self.search_box)

        self.add_button = QtWidgets.QPushButton("Add Selected")
        self.add_button.clicked.connect(self._on_add_structure)
        self.add_button.setEnabled(False)
        atlas_row.addWidget(self.add_button)

        layout.addLayout(atlas_row)

        # Structure List
        self.structure_list = QtWidgets.QListWidget()
        self.structure_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        layout.addWidget(self.structure_list)

        # Status Label
        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)


    def _load_default_atlas(self) -> None:
        """
        Load the default atlas on initialization in the combobox.
        """
        self._on_atlas_changed(0)


    def _on_atlas_changed(self, index: int) -> None:
        """
        Handle atlas selection changes.
        :param atlas_name: Selected atlas name.
        """
        
        atlas_name = self.atlas_combo.currentText()
        
        if not atlas_name:
            return 
        
        try: 
            self.status_label.setText(f"Loading atlas '{atlas_name}'...")
            QtWidgets.QApplication.processEvents()

            structures_df = self.atlas_service.load_atlas(atlas_name)
            self._update_structure_list(structures_df)
            self.status_label.setText(f"Atlas '{atlas_name}' loaded with {len(structures_df)} structures.")
            self.add_button.setEnabled(True)
        
        except ValueError as e:
            self.status_label.setText(str(e))
            self.structure_list.clear()
            self.add_button.setEnabled(False)

    
    def _on_search_changed(self, query: str) -> None:
        """
        Handle search box text changes.
        :param query: Search query string.
        """
        structures_df = self.atlas_service.search_structure(query)
        self._update_structure_list(structures_df)


    def _on_search_text_changed(self, text: str) -> None:
        """
        Handle search box text changes.
        :param text: Search query string.
        """
        filtered_df = self.atlas_service.search_structure(text)
        self._update_structure_list(filtered_df)

    
    def _update_structure_list(self, filtered_df) -> None:
        """
        Update the structure list widget with filtered structures.
        :param filtered_df: DataFrame of filtered structures.
        """
        self.structure_list.clear()

        for _, row in filtered_df.iterrows():
            item = QtWidgets.QListWidgetItem(f"{row['acronym']}: {row['name']}")
            item.setData(QtCore.Qt.UserRole, row['acronym'])
            self.structure_list.addItem(item)


    def _on_add_structure(self) -> None:
        """
        Handle the "Add Selected" button click.
        Emit the selected structure acronyms.
        """
        selected_items = self.structure_list.selectedItems()

        if not selected_items:
            return 
        
        mesh_paths = []
        errors = [] 

        for item in selected_items:
            acro = item.data(QtCore.Qt.UserRole)

            try:
                mesh_path = self.atlas_service.get_mesh_path(acro)
                mesh_paths.append(mesh_path)
            except ValueError as e:
                errors.append(str(e))

            
        if mesh_paths:
            self.structures_selected.emit(mesh_paths) 

            self.status_label.setText(
                f"Added {len(mesh_paths)} structures.")
            
        if errors:
            self.status_label.setText(
                f"Errors: {'; '.join(errors)}")

