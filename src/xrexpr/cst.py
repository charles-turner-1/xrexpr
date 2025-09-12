import libcst as cst


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
