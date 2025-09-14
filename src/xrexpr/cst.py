import libcst as cst


class InvalidExpressionError(Exception):
    pass


class SelectionPushdown(cst.CSTTransformer):
    """
    Pushes isel calls down to the first position in a chain of mean calls, by
    matching on node and exchanging pairs
    """

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        match updated_node:
            case cst.Call(
                func=cst.Attribute(
                    value=cst.Call(
                        func=cst.Attribute(
                            value=base_val,
                            attr=cst.Name(
                                value="mean",
                            ),
                        ),
                        args=mean_args,
                    ),
                    attr=cst.Name(
                        value=selector,
                    ),
                ),
                args=isel_args,
            ) if selector in ["isel", "sel"]:
                # Check for any overlap in mean kwarg values, and selector kwarg names
                # if self._not_reorderable(mean_args, isel_args):
                #     return updated_node

                swapped_node = cst.Call(
                    func=cst.Attribute(
                        value=cst.Call(
                            func=cst.Attribute(
                                value=base_val,
                                attr=cst.Name(
                                    value=selector,
                                ),
                            ),
                            args=isel_args,
                        ),
                        attr=cst.Name(
                            value="mean",
                        ),
                    ),
                    args=mean_args,
                )
                return swapped_node.visit(self)  # type: ignore[return-value]

        return updated_node

    def _not_reorderable(
        self, mean_args: list[cst.Arg], isel_args: list[cst.Arg]
    ) -> bool:
        """
        This is absolutely gross and needs refactoring, to extract the simplestring value.

        If we cannot reorder, then we actually want to raise - the selection will
        be invalid
        """

        mean_dims = {key.value.value.strip("'").strip('"') for key in mean_args}
        isel_dims = {key.keyword.value for key in isel_args if key.keyword}

        not_reorderable = not bool(mean_dims & isel_dims)

        return not_reorderable

        if not_reorderable:
            return True

        raise InvalidExpressionError(
            "Expression not valid: selection on dropped dimensions"
        )
